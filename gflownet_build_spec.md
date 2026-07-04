# Fragment GFlowNet — build spec (phase 1)

Locked-in choices unchanged: fragment-by-fragment construction, fragment-level GNN state encoder, Trajectory Balance, fixed-weight scalar reward, torchgfn + PyG + RDKit + ZINC250K/BRICS. Below are the six spec decisions needed to start writing code.

Dims used throughout: `d = 128` (node embedding, graph embedding, block embedding all 128). ECFP: radius 2, 2048-bit. GNN: 3 layers, GINE (edge-aware). One base-fragment vocab `V ≈ 200`, expanded vocab `V' ≈ 300–500`.

---

## 1. State representation (code-level)

**Decision.** A partial molecule is a *fragment graph*: nodes are placed fragments, edges are the bonds joining them at specific attachment points. Keep two synchronized views:

- **Python (source of truth for stepping / RDKit assembly)** — a frozen dataclass:
  ```python
  @dataclass(frozen=True)
  class FragState:
      frags: tuple[int, ...]                       # base-vocab id per placed fragment, in placement order
      bonds: tuple[tuple[int, int, int, int], ...] # (frag_i, ap_i, frag_j, ap_j), sorted canonically
  ```
  An attachment point `(frag, ap)` is *open* iff it appears in no bond. Terminal molecule is built from this by RDKit at Stop (remove BRICS dummy atoms, form real bonds, sanitize).

- **Tensor view for torchgfn / the GNN** — a PyG `Data`:
  - `x`: `[N, F_in]` node features = fragment's ECFP row (`F_in = 2048`), later projected to `d` inside the model. Append an open-AP multi-hot over `max_AP = 4` slots → node feature carries which of its APs are still free. So `F_in = 2048 + 4`.
  - `edge_index`: `[2, 2E]` (both directions), `edge_attr`: `[2E, 8]` = one-hot of `(ap_i, ap_j)` slot indices (`2 * max_AP`).
  - Wrap in torchgfn's graph-state support (`GraphStates`).

**Why.** Fragment-level nodes match the BRICS decomposition and the GNN choice directly — no atom-level bookkeeping. The dataclass is hashable/comparable (torchgfn needs this) and trivially invertible for the backward step; the PyG view is what the GNN consumes. Splitting the two avoids forcing RDKit mol objects through torchgfn's tensor plumbing.

**Concrete / caveats.**
- `max_AP = 4` (fragments with >4 attachment points are filtered out of the vocab, see §3), so all per-AP tensors are fixed-width.
- **Order redundancy (accept it in phase 1):** two build orders reaching the same molecule are two distinct `FragState`s. That's correct for a GFlowNet DAG — reward is computed on the RDKit canonical SMILES at the terminal, and the sampler draws each molecule ∝ the summed flow over its terminal representations. Don't attempt graph canonicalization now.
- **torchgfn version caveat:** if graph-state support is finicky in your installed version, the drop-in fallback is a padded tensor encoding — `frag_ids [N_max]` (pad id for empty slots) + `adj [N_max, N_max, max_AP]` — with `N_max = 8`. Same masks, same policy heads; only the encoder input changes.

---

## 2. Action space enumeration

**Decision.** Two action types: **Stop**, and **Add** = attach one expanded-vocab entry at one open attachment point of the current molecule. The Add action is the pair `(focus_ap, block)` where `focus_ap` is an open AP on the current graph and `block` is an expanded-vocab entry (§3). Factor the forward policy as one categorical whose logits are computed from factored pieces, so you never materialize the full dense space:

```
forward actions from state s  =  {STOP}  ∪  {(focus_ap, block) : focus_ap ∈ open_APs(s), block ∈ V'}
logits:  stop_logit          = MLP_stop(h_G)                       # scalar
         add_logit(a, e)     = MLP_add([ h_ap[a] ; b_e ; h_G ])    # per open-AP a, per block e
size:    1 + |open_APs(s)| · V'
```

- `|open_APs(s)|` is small in practice (≤ ~12), so the categorical is a few thousand entries, not `V' × N_max × max_AP`.
- **Source step is special:** from the empty state, Add just picks a *base* fragment id (no focus AP; all its APs open). Categorical of size `V`.
- **Masking:** invalid `(focus_ap, block)` pairs are masked — an AP already bonded, or an `N_max`-cap violation. Stop is masked off until ≥1 fragment is placed (no empty molecule as output).

**Why.** The naive space `open_APs × V' × new_frag_AP` is huge; two collapses fix it. (a) Pre-expanding the vocab so each block already fixes *its own* attachment point (§3) removes the third dimension entirely. (b) Computing logits from `[AP-embedding ; block-embedding ; graph-embedding]` means you only ever score the currently-open APs, not a padded max — cost scales with the actual molecule, and it stays a single clean categorical torchgfn can step through.

**Concrete.** `N_max = 8` fragments cap. `MLP_add`: `[d + d + d] → 64 → 1`. `MLP_stop`: `[d] → 64 → 1`.

---

## 3. Fragment vocabulary construction

**Decision.** BRICS-decompose all of ZINC250K, canonicalize fragments (canonical SMILES with dummy atoms kept), count occurrences, then filter and take the top-K:

1. Drop fragments with >4 attachment points (dummy atoms) — bounds `max_AP`.
2. Drop fragments outside 2–20 heavy atoms (junk / oversized).
3. Keep the **top 200 by frequency** → base vocab `V ≈ 200`.
4. **Expand:** for each base fragment, emit one entry per attachment point → `block = (fragment, connecting_ap)`. `V' ≈ 300–500`.

**Why.** BRICS on 250k molecules yields tens of thousands of unique fragments with a heavy long tail; the top few hundred cover the large majority of occurrences, so a top-200 cut keeps chemistry coverage while making the action categorical (§2) small. Filtering AP count and size keeps tensors fixed-width and removes fragments that blow up the branching factor or produce implausible pieces. Expanding the vocab so each entry pins its own attachment point is what lets the Add action drop a whole dimension.

**Concrete.** Cutoff is "top-200 by count" (equivalently ~appears in ≥ ~0.1% of molecules); treat 200 as the tunable knob — bump to 300–400 if coverage feels thin. Precompute and cache: base SMILES list, AP counts per fragment, and the expanded `(fragment, ap)` table.

---

## 4. Fragment embeddings

**Decision.** Fixed **ECFP4 (Morgan, radius 2, 2048-bit) per fragment**, precomputed once into a `[V', 2048]` table and projected to `d` by a *learned* linear layer inside the model. Not a pretrained per-fragment GNN.

**Why.** The state encoder is already a fragment-level GNN whose job is to reason over how fragments are wired together; each node only needs a feature that *identifies the fragment's chemistry*, which a fixed substructure fingerprint does well for a few-hundred-entry vocab. A pretrained per-fragment GNN adds an entire second training stage (objective, data, checkpointing) for marginal gain at this vocab size — it violates "as simple as possible." The fingerprint is deterministic, zero-training, and the learned projection lets the model adapt it end-to-end.

**Concrete.**
- `frag_ecfp: FloatTensor[V', 2048]`, built with RDKit `GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)`.
- Projection `Linear(2048, d=128)` — shared between the state-GNN node inputs and the block embeddings `b_e` used in §2.
- (Noted alternative, not chosen: a learnable `nn.Embedding(V', d)` is even simpler but throws away chemical structure and can't warm-start unseen fragments — fingerprint is the better fixed default. Easy to swap if you want to A/B it.)

---

## 5. Policy network I/O shapes

Shared encoder: GINE, 3 layers, hidden `d = 128`.
- **In:** `x [N, 2052]` (2048 ECFP + 4 open-AP multi-hot), `edge_index [2, 2E]`, `edge_attr [2E, 8]`.
- **Out:** node embeddings `h_v [N, 128]`; graph embedding `h_G [1, 128]` via mean+add pooling (`[N,128] → [256] → Linear → [128]`).
- Per-AP embedding: `h_ap[a] = MLP([ h_v[frag(a)] ; slot_emb[ap_slot(a)] ])`, where `slot_emb = nn.Embedding(max_AP=4, 32)`. Shape `[num_open_AP, 128]`.

**Forward policy.**
- In: `h_v, h_G`, block table `b_e [V', 128]`.
- Out: log-probs over `1 + num_open_AP · V'` actions (masked). Stop logit `MLP_stop(h_G)`; add logits `MLP_add([h_ap[a]; b_e; h_G])` → `[num_open_AP, V']`, flatten, concat Stop, log-softmax over valid entries.
- Source-state head: `Linear(d → V)` on `h_G` of the empty-graph embedding (a learned constant vector) → categorical over base fragments.

**Backward policy.**
- In: `h_v` from the same (or a separate) encoder.
- Out: categorical over *current fragments*, one logit each `MLP_back(h_v) → [N, 1]`, masked to **removable leaves** (fragments of degree 1 in the fragment graph). Removing a leaf reopens the neighbor's AP and is the exact inverse of the Add that placed it.
- Single-fragment state → source is the only parent → deterministic (`P_B = 1`).

**Z.** One global learned scalar `log_Z: nn.Parameter(torch.zeros(1))`, per Trajectory Balance.

**Why.** Node/AP-level logits for "where to attach," graph-level logit for Stop, and a per-fragment logit for "what to undo" mirror the MDP's actual decision structure and keep every head bounded by the current molecule size. Backward-over-leaves gives exactly the parent set (each leaf removal = one distinct parent), so `P_B` is a proper distribution over predecessors and TB is consistent.

---

## 6. Reward function specifics

**Decision.** Normalize each objective to a [0,1] "goodness," combine with fixed weights, then apply the GFlowNet reward exponent. Computed only at Stop, on the sanitized RDKit molecule.

```
q = QED(mol)                              # already in [0,1], higher better
s = clip((10 - SAscore(mol)) / 9, 0, 1)   # RDKit sascorer ~[1,10], lower better → map to [0,1]
l = exp(-0.5 * ((LogP(mol) - 2.5) / 2.0)**2)  # Gaussian window, peak at target LogP 2.5

R_raw = 0.5*q + 0.3*s + 0.2*l             # weights sum to 1 → R_raw ∈ [0,1]
logR  = beta * log(R_raw + 1e-4)          # TB works in log-space
```

**Why.** The three signals live on different scales (QED bounded, SA is 1–10 and inverted, LogP unbounded and *not* monotone-good), so each needs its own mapping before they can share one scalar. QED gets the largest weight (overall drug-likeness), SA next (synthesizability is the point of a fragment build), LogP least and as a soft window — a Gaussian centered at 2.5 rewards drug-like lipophilicity without hard-rejecting, which is gentler than a Lipinski step and keeps the reward smooth for exploration.

**Concrete.**
- Weights `(w_q, w_s, w_l) = (0.5, 0.3, 0.2)`; LogP target `μ = 2.5`, `σ = 2.0`.
- `beta = 1` for the first working run (phase 1 = don't sharpen); it's the main exploration↔exploitation knob, tune up to ~4–8 later.
- `ε = 1e-4` floor so `log` is finite.
- **Invalid molecule** (RDKit build/sanitize fails, or disconnected graph at Stop) → `R_raw = 0` → `logR = log(ε)`, strongly discouraged.
- If you'd rather LogP be monotone (reward lower, Lipinski-style) than windowed, swap `l = sigmoid(5 - LogP)` — flagged as the one genuinely open modeling choice here; the windowed default is my recommendation.

---

### Cross-cutting notes
- Only §6's LogP shaping and §3's cutoff-of-200 are "tune-me" knobs; everything else is a committed decision.
- No objective/state-encoding/vocab-strategy changes were introduced beyond the six items.
- First milestone to validate the plumbing before rewards matter: overfit a tiny fixed reward on `V ≈ 20` fragments, `N_max = 4`, and confirm the TB loss drives `log_Z` toward `log Σ_x R(x)`.
