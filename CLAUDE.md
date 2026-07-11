# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FragFlow** is a fragment-based GFlowNet for de novo molecular generation. It constructs molecules incrementally by assembling molecular fragments using Generative Flow Networks (GFlowNets) with Trajectory Balance objective. The system uses BRICS decomposition on ZINC250K to build a fragment vocabulary, then learns to assemble these fragments into drug-like molecules optimized for QED, synthetic accessibility (SA), and LogP.

**Core stack:** torchgfn + PyTorch Geometric (PyG) + RDKit + ZINC250K dataset

## Architecture

### State Representation (Dual View)

The system maintains **two synchronized representations** of partial molecules:

1. **Python source of truth** (`env/molecule_state.py`):
   ```python
   @dataclass(frozen=True)
   class FragState:
       frags: tuple[int, ...]                       # base-vocab id per fragment, placement order
       bonds: tuple[tuple[int, int, int, int], ...] # (frag_i, ap_i, frag_j, ap_j)
   ```
   - Frozen/hashable for torchgfn compatibility
   - Used for stepping logic and RDKit assembly at terminal states
   - Attachment points (APs) are "open" if they appear in no bond tuple

2. **Tensor view for GNN** (PyG `Data` wrapped in torchgfn `GraphStates`):
   - Nodes = placed fragments
   - Node features: `[N, 2052]` = ECFP4 fingerprint (2048-bit) + 4-slot open-AP multi-hot
   - Edges = bonds between fragments at specific APs
   - Edge attributes: `[2E, 8]` = one-hot of `(ap_i, ap_j)` slot indices

**Critical invariant:** Fragment graphs with different build orders are distinct states in the GFlowNet DAG. Reward computed on canonical SMILES at terminal ensures correct sampling distribution (molecules sampled ∝ summed flow over their terminal representations).

### Action Space (`env/actions.py`)

Two action types:
- **Stop**: Terminate and assemble final molecule via RDKit
- **Add**: `(focus_ap, block)` where `focus_ap` is an open AP on current molecule, `block` is an expanded-vocab entry

The forward policy **factors** the action space to avoid materializing `|open_APs| × V' × max_AP`:
```
stop_logit = MLP_stop(h_G)                    # scalar
add_logit(a, e) = MLP_add([h_ap[a]; b_e; h_G])  # per open-AP a, per block e
```
Total categorical size: `1 + |open_APs(s)| · V'` (typically ~few thousand)

**Special cases:**
- Source step (empty state): picks a base fragment (categorical size `V ≈ 200`)
- Masking: invalid pairs masked; Stop masked until ≥1 fragment placed

### Fragment Vocabulary (`data/`)

**Base vocab construction** (run once, cached in `data/fragments/`):
1. BRICS-decompose ZINC250K
2. Filter: drop fragments with >4 APs or outside 2-20 heavy atoms
3. Keep top-200 by frequency → base vocab `V ≈ 200`
4. **Expand:** emit one entry per (fragment, attachment_point) → `V' ≈ 300-500`

Why expand? Pre-pinning the attachment point collapses `(AP, fragment, new_AP)` action space to just `(AP, block)`, removing a dimension.

### Models Architecture

**Fragment embeddings** (`models/fragment_embed.py`):
- Fixed ECFP4 (Morgan radius=2, 2048-bit) per fragment → `[V', 2048]` table
- Learned linear projection: `Linear(2048, d=128)` used for both state-GNN inputs and block embeddings

**State encoder** (`models/state_encoder.py`):
- GINE (edge-aware GNN), 3 layers, hidden dim `d=128`
- Input: `x [N, 2052]`, `edge_index [2, 2E]`, `edge_attr [2E, 8]`
- Output: node embeddings `h_v [N, 128]`, graph embedding `h_G [1, 128]` (mean+add pooling)
- Per-AP embedding: `h_ap[a] = MLP([h_v[frag(a)]; slot_emb[ap_slot(a)]])` where `slot_emb` is learned 4-entry table

**Forward policy** (`models/forward_policy.py`):
- Stop head: `MLP_stop(h_G) → scalar`
- Add head: `MLP_add([h_ap[a]; b_e; h_G]) → [num_open_AP, V']` then flatten
- Source head (empty state): `Linear(h_G → V)` over base fragments

**Backward policy** (`models/backward_policy.py`):
- `MLP_back(h_v) → [N, 1]` per fragment node
- Masked to **removable leaves** (degree-1 fragments in fragment graph)
- Removing a leaf is exact inverse of the Add that placed it
- Single-fragment state → deterministic parent (source)

**Learned parameter:** `log_Z: nn.Parameter(torch.zeros(1))` for Trajectory Balance

### Reward Function (`reward/reward_fn.py`)

Computed only at terminal (Stop), on sanitized RDKit molecule:

```python
q = QED(mol)                              # [0,1], drug-likeness
s = clip((10 - SAscore(mol)) / 9, 0, 1)   # SA inverted to [0,1]
l = exp(-0.5 * ((LogP(mol) - 2.5) / 2.0)**2)  # Gaussian window, peak at 2.5

R_raw = 0.5*q + 0.3*s + 0.2*l             # weighted combination
logR = beta * log(R_raw + 1e-4)           # TB log-space reward
```

**Key parameters:**
- Weights: `(w_QED, w_SA, w_LogP) = (0.5, 0.3, 0.2)`
- LogP target: `μ=2.5`, `σ=2.0`
- `beta=1` for initial training (exploration knob, tune to 4-8 later)
- Invalid molecule → `R_raw=0` → `logR=log(1e-4)` (strongly discouraged)

### Training (`training/`)

**Loss:** Trajectory Balance (TB) from torchgfn
- Samples trajectories from current policy
- Updates `P_F`, `P_B`, and `log_Z` to satisfy flow-matching constraint
- Config: `training/config.py` (learning rate, batch size, replay buffer, wandb settings)
- Main loop: `training/train.py`

## Fixed Design Constraints

These are **committed decisions**, not tunable knobs:
- `d=128`: all embeddings (node, graph, block)
- `max_AP=4`: fragments with >4 attachment points filtered from vocab
- `N_max=8`: maximum fragments per molecule
- ECFP4 (radius=2, 2048-bit) for fragment features
- GINE with 3 layers for state encoder
- Trajectory Balance objective (not subtb, flow-matching, etc.)
- Backward policy over removable leaves (degree-1 fragments)

## Tunable Parameters

- Fragment vocab size cutoff (default: top-200, expandable to 300-400)
- Reward `beta` (exploration ↔ exploitation, start at 1, increase to 4-8)
- LogP shaping: Gaussian window (current) vs monotone sigmoid
- Training hyperparameters in `training/config.py`

## Module Responsibilities

- **`data/`**: ZINC250K loading, BRICS fragmentation, vocab caching
- **`env/`**: State representation (`FragState` ↔ PyG `Data`), action enumeration, stepping logic, terminal check
- **`models/`**: Fragment embeddings (ECFP projection), state encoder (GINE), forward/backward policy heads
- **`reward/`**: QED/SA/LogP → scalar reward, RDKit sanitization handling
- **`training/`**: TB loss, sampling, optimizer, wandb logging
- **`evaluation/`**: Sample generation, diversity/validity metrics

## First Milestone

Before full training, **overfit a sanity check:**
- Tiny vocab (`V ≈ 20`), `N_max=4`, fixed dummy reward
- Confirm TB loss drives `log_Z → log Σ_x R(x)`
- Validates plumbing (state stepping, policy masking, reward computation) before chemistry matters
