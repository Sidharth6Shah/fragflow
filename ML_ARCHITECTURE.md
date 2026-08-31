# FragFlow - Complete ML Architecture & Workflow

**Author:** Generated for deep-dive ML walkthrough
**Date:** 2024-08-28
**Model:** Trained GFlowNet for fragment-based molecular generation

---

## Overview

**FragFlow** is a fragment-based molecular generation system using **Generative Flow Networks (GFlowNets)** with the **Trajectory Balance (TB)** objective. It learns to assemble drug-like molecules from molecular fragments optimized for QED (drug-likeness), synthetic accessibility (SA), and LogP (lipophilicity).

**Core Innovation:** Instead of generating molecules atom-by-atom or with SMILES strings, FragFlow builds molecules by assembling pre-validated molecular fragments (from BRICS decomposition), making generation more chemically valid and controllable.

---

## 1. Problem Formulation

### Goal
Learn a generative policy that samples molecular structures `x` with probability proportional to a reward function `R(x)`:

```
p(x) ∝ R(x)
```

Where `R(x)` combines:
- **QED** (0-1): Drug-likeness
- **SA Score** (1-10, lower better): Synthetic accessibility
- **LogP**: Lipophilicity (Gaussian window around 2.5)

### Why GFlowNets?
Traditional generative models (VAEs, GANs) struggle with:
1. **Mode collapse** - generating only a few "safe" molecules
2. **Invalid structures** - violating chemical rules
3. **Diversity-quality tradeoff** - hard to get both high reward AND diversity

**GFlowNets solve this by:**
- Sampling proportionally to reward (not just maximizing)
- Naturally exploring diverse high-reward modes
- Learning through flow-matching instead of direct likelihood maximization

---

## 2. State Representation (Dual View)

FragFlow maintains **two synchronized representations** of partial molecules:

### 2.1 Python Source of Truth (`env/molecule_state.py`)

```python
@dataclass(frozen=True)
class FragState:
    frags: tuple[int, ...]                       # Fragment IDs in placement order
    bonds: tuple[tuple[int, int, int, int], ...] # (frag_i, ap_i, frag_j, ap_j)
```

**Example:**
```python
frags = (42, 17, 89)  # 3 fragments placed
bonds = ((0, 1, 1, 0), (1, 2, 2, 3))  # Fragment 0's AP 1 → Fragment 1's AP 0
                                        # Fragment 1's AP 2 → Fragment 2's AP 3
```

**Key Properties:**
- Frozen/hashable (required by torchgfn)
- Build order matters (different orders = different states in GFlowNet DAG)
- Attachment points (APs) are "open" if they appear in no bond tuple

### 2.2 Graph Neural Network View (`env/molecule_state.py:state_to_pyg()`)

Converted to PyTorch Geometric `Data` for GNN processing:

```python
Data:
    x: [N, 2052]           # Node features per fragment
    edge_index: [2, 2E]    # Bonds between fragments
    edge_attr: [2E, 8]     # Edge features (AP indices)
```

**Node features (2052-dim):**
- **2048-bit ECFP4 fingerprint** (Morgan radius=2) of fragment
- **4-bit open-AP multi-hot** (max 4 APs per fragment)
  - Example: `[1, 0, 1, 0]` = APs 0 and 2 are open

**Edge attributes (8-dim):**
- One-hot encoding of `(ap_i, ap_j)` slot indices
- Example: bond from AP 1 of fragment A to AP 0 of fragment B → `[0,1,0,0, 1,0,0,0]`

---

## 3. Action Space

### 3.1 Action Types

1. **Stop**: Terminate trajectory, assemble final molecule
2. **Add(focus_ap, block)**: Add a new fragment
   - `focus_ap`: Open attachment point on current molecule
   - `block`: Entry from expanded vocabulary (fragment + pinned AP)

### 3.2 Expanded Vocabulary

**Base vocab construction** (`data/prepare_data.py`):
1. BRICS-decompose ZINC250K dataset
2. Filter: keep fragments with 2-20 heavy atoms, ≤4 attachment points
3. Top-200 by frequency → **base vocab** `V ≈ 200`

**Expansion** (critical for action space reduction):
4. For each fragment, create one entry per attachment point
   - Example: Fragment with 3 APs → 3 entries in expanded vocab
   - **Expanded vocab** `V' ≈ 226` (actual size from training)

**Why expand?**
Pre-pinning the attachment point collapses the action space from `(AP, fragment, new_AP)` to just `(AP, block)`, removing one dimension.

### 3.3 Action Enumeration

**Source step** (empty state):
- Categorical over base fragments: size `V ≈ 200`

**Regular step**:
- Stop: 1 action
- Add: `|open_APs| × V'` combinations
- Total: `1 + |open_APs| × V'` (typically ~few thousand)

**Masking:** Invalid pairs masked; Stop masked until ≥1 fragment placed

---

## 4. Neural Network Architecture

All embeddings use dimension `d = 128` (fixed design constraint).

### 4.1 Fragment Embeddings (`models/fragment_embed.py`)

```python
FragmentEmbedding:
    ecfp_table: [V', 2048]  # Pre-computed ECFP4 fingerprints
    projection: Linear(2048 → 128)
```

Returns `[V', 128]` learned embeddings for all expanded vocab entries.

### 4.2 State Encoder (`models/state_encoder.py`)

**GINE (Graph Isomorphism Network with Edge features):**
- 3 layers, hidden dim 128
- Edge-aware message passing

```python
StateEncoder:
    input_proj: Linear(2052 → 128)    # Node features
    edge_proj: Linear(8 → 128)        # Edge features
    convs: [GINE × 3]                 # Message passing layers
    ap_slot_emb: Embedding(4, 128)    # Learnable AP slot embeddings
```

**Forward pass:**
1. Project node features: `h_v^0 = input_proj(x)`
2. Project edge features: `e = edge_proj(edge_attr)`
3. Message passing (3 layers):
   ```python
   h_v^{l+1} = MLP([h_v^l; Σ_{u∈N(v)} MLP(h_u^l + e_{uv})])
   ```
4. Graph pooling: `h_G = mean(h_v) + sum(h_v)`

**Per-AP embeddings** (for action selection):
```python
h_ap[a] = MLP([h_v[frag(a)]; ap_slot_emb[ap_slot(a)]])
```

### 4.3 Forward Policy (`models/forward_policy.py`)

**Factorized action space:**

```python
ForwardPolicy:
    stop_head: MLP(h_G → scalar)
    add_head: MLP([h_ap; block_emb; h_G] → scalar)
    source_head: Linear(h_G → V)
```

**At source step:**
```python
logits = source_head(h_G)  # [V] categorical over base fragments
```

**At regular step:**
```python
stop_logit = stop_head(h_G)  # scalar
for each open AP a, block e:
    add_logit[a, e] = add_head([h_ap[a]; block_emb[e]; h_G])
# Concatenate: [stop_logit, add_logits.flatten()]
# Total size: 1 + |open_APs| × V'
```

### 4.4 Backward Policy (`models/backward_policy.py`)

**Leaf removal policy:**

```python
BackwardPolicy:
    mlp: MLP(h_v → scalar)
```

- Computes logits over **removable leaves** (degree-1 fragments in fragment graph)
- Removing a leaf is exact inverse of the Add that placed it
- Single-fragment state → deterministic parent (source)

### 4.5 Learned Flow Parameter

```python
log_Z: nn.Parameter(torch.zeros(1))
```

Learned partition function estimate for Trajectory Balance.

---

## 5. Reward Function (`reward/reward_fn.py`)

Computed **only at terminal states** (after Stop action).

### 5.1 Full Reward Formula

```python
def compute_full_reward(mol, beta=1.0):
    # QED: [0, 1], higher better
    q = QED.qed(mol)

    # SA Score: [1, 10], lower better → invert to [0, 1]
    sa_score = sascorer.calculateScore(mol)
    s = clip((10 - sa_score) / 9.0, 0, 1)

    # LogP: Gaussian window around target 2.5
    logp = Descriptors.MolLogP(mol)
    l = exp(-0.5 * ((logp - 2.5) / 2.0)^2)

    # Weighted combination
    R_raw = 0.5*q + 0.3*s + 0.2*l

    # Log-space reward for TB
    logR = beta * log(R_raw + 1e-4)

    return logR
```

**Parameters:**
- Weights: `(QED, SA, LogP) = (0.5, 0.3, 0.2)`
- LogP target: `μ=2.5`, `σ=2.0`
- `beta=1.0` (trained with this, can tune to 4-8 for more exploration)
- Invalid molecule → `R_raw=0` → `logR=log(1e-4)` (strongly discouraged)

### 5.2 Scaffold Reward (Optional)

```python
def compute_scaffold_reward(mol, beta, scaffold_smarts, scaffold_weight=0.3):
    base_reward = compute_full_reward(mol, beta)  # QED + SA + LogP

    # Check scaffold match
    scaffold_mol = Chem.MolFromSmarts(scaffold_smarts)
    scaffold_bonus = 1.0 if mol.HasSubstructMatch(scaffold_mol) else 0.0

    # Combine
    R_raw = (1 - scaffold_weight)*base_reward + scaffold_weight*scaffold_bonus
    return beta * log(R_raw + 1e-4)
```

**Note:** Model was trained with `mode="full"`, not scaffold mode. Scaffold filtering works post-generation.

---

## 6. Training: Trajectory Balance (TB)

### 6.1 Trajectory Sampling

Sample complete trajectories from source to terminal:

```python
τ = (s_0 → s_1 → ... → s_T)
```

At each step `t`:
1. Encode state: `h_v, h_G = encoder(state_to_pyg(s_t))`
2. Compute forward policy: `P_F(a | s_t)`
3. Sample action: `a_t ~ P_F(· | s_t)`
4. Step environment: `s_{t+1}, done = env.step(a_t)`
5. If done, compute reward: `R(s_T)`

### 6.2 Trajectory Balance Loss

**Core equation:**
```
Z × ∏_t P_F(a_t | s_t) = R(s_T) × ∏_t P_B(a_t | s_{t+1})
```

Where:
- `Z`: Partition function (sum of rewards over all terminal states)
- `P_F`: Forward policy
- `P_B`: Backward policy
- `R(s_T)`: Reward at terminal state

**Loss (log-space):**
```python
loss = (log_Z + Σ_t log P_F(a_t|s_t) - Σ_t log P_B(a_t|s_{t+1}) - log R(s_T))^2
```

**Gradients update:**
- `log_Z` parameter
- Forward policy parameters (encoder, fragment embeddings, policy heads)
- Backward policy parameters

### 6.3 Training Configuration (`training/config.py`)

**Full training config:**
```python
batch_size = 16              # Trajectories per update
lr = 1e-3                    # Adam learning rate
max_frags = 8                # Maximum fragments per molecule
beta = 1.0                   # Reward temperature
vocab_size = 200             # Base vocabulary size
max_iterations = 10000       # Training steps
grad_clip = 1.0              # Gradient clipping max norm
```

**Optimizer:**
```python
Adam(params, lr=1e-3)
# Gradient clipping: torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

---

## 7. Training Workflow

### 7.1 Data Preparation

```bash
python data/prepare_data.py
```

1. Download ZINC250K dataset
2. BRICS decomposition of all molecules
3. Filter fragments: 2-20 heavy atoms, ≤4 APs
4. Count frequencies, keep top-200
5. Expand vocabulary (fragment × AP combinations)
6. Compute ECFP4 fingerprints for all entries
7. Cache to `data/fragments/vocab_top200.pkl`

### 7.2 Training Loop (`training/train.py`)

```python
for iteration in range(max_iterations):
    # Sample batch of trajectories
    trajectories = []
    for _ in range(batch_size):
        trajectory = sample_trajectory(env, forward_policy, encoder, frag_embed, vocab)
        trajectories.append(trajectory)

    # Compute TB loss
    loss = 0
    for τ in trajectories:
        log_pf = sum(log_prob(a_t | s_t) for t in τ)
        log_pb = sum(log_prob(a_t | s_{t+1}) for t in τ)
        log_r = log_reward(τ.terminal_state)

        loss += (log_Z + log_pf - log_pb - log_r)^2

    loss = loss / batch_size

    # Backward pass
    loss.backward()
    clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Log metrics
    if iteration % 100 == 0:
        wandb.log({
            'loss': loss.item(),
            'log_Z': log_Z.item(),
            'avg_reward': mean(rewards),
            'avg_fragments': mean(num_fragments)
        })
```

### 7.3 Actual Training Results

**Platform:** Lambda Labs GPU (A100 40GB)
**Duration:** ~16 hours for 10K iterations
**Cost:** ~$32 ($1.99/hr)

**Convergence:**
- Loss: 0.0995 → 0.0055
- Avg reward: -1.29 → -1.05
- log_Z: 0.0 → 8.46

**Final checkpoint:** `checkpoints/checkpoint_10000.pt`

---

## 8. Evaluation (`evaluation/sample.py`, `evaluation/metrics.py`)

### 8.1 Sampling

```python
sampler = MoleculeSampler("checkpoints/checkpoint_10000.pt")
molecules = sampler.sample_batch(num_samples=1000)
```

Uses greedy or stochastic sampling from forward policy.

### 8.2 Metrics (1000 molecules)

| Metric | Value | Meaning |
|--------|-------|---------|
| **Validity** | 100% | All molecules chemically valid |
| **Uniqueness** | 39% | Distinct molecules / total |
| **Diversity** | 81.2% | Avg Tanimoto distance |
| **Mean Reward** | -0.316 | Log-space reward |
| **Mean Fragments** | 2.1 | Avg fragments per molecule |

**Fragment Distribution:**
- 2 fragments: 93.0%
- 3 fragments: 5.4%
- 4-6 fragments: 1.6%
- 0% single-fragment (critical success - mode collapse was bug)

---

## 9. Critical Bug Fix (molecule_state.py:284)

### 9.1 The Bug

**Original code (WRONG):**
```python
# Count dummies AFTER index i
removed_before = sum(1 for d in sorted_dummies if d > i)
```

**Fixed code:**
```python
# Count dummies BEFORE index i
removed_before = sum(1 for d in sorted_dummies if d < i)
```

### 9.2 Impact

**Before fix:**
- 100% mode collapse to single-fragment molecules
- Multi-fragment assembly broken
- Mean reward: -3.7

**After fix:**
- 0% single-fragment molecules
- 93% 2-fragment, 5.4% 3-fragment
- Mean reward: -0.316 (10x better)

This bug caused incorrect index mapping when removing dummy atoms during molecule assembly, making all multi-fragment molecules invalid. The model learned to avoid multi-fragment assembly entirely.

---

## 10. Key Design Constraints (Fixed, Not Tunable)

These are **committed architectural decisions:**

1. **d=128**: All embeddings (node, graph, block)
2. **max_AP=4**: Fragments with >4 attachment points filtered
3. **N_max=8**: Maximum fragments per molecule
4. **ECFP4**: Morgan fingerprint radius=2, 2048-bit
5. **GINE 3-layer**: Graph neural network architecture
6. **Trajectory Balance**: Not SubTB, Flow-Matching, etc.
7. **Backward = leaf removal**: Remove degree-1 fragments only

---

## 11. File Structure

```
fragflow/
├── data/
│   ├── prepare_data.py          # Vocab construction
│   ├── fragments/
│   │   └── vocab_top200.pkl     # Cached vocabulary
│   └── raw/                     # ZINC250K download
│
├── env/
│   ├── molecule_state.py        # FragState, state_to_pyg(), state_to_molecule()
│   ├── actions.py               # Action enumeration, masking
│   └── molecule_env.py          # Environment step(), reset(), is_terminal()
│
├── models/
│   ├── fragment_embed.py        # ECFP → learned embeddings
│   ├── state_encoder.py         # GINE encoder
│   ├── forward_policy.py        # P_F (Stop/Add/Source heads)
│   └── backward_policy.py       # P_B (leaf removal)
│
├── reward/
│   ├── reward_fn.py             # QED/SA/LogP → scalar reward
│   └── docking_reward.py        # Placeholder for protein binding
│
├── training/
│   ├── config.py                # Hyperparameters
│   └── train.py                 # TB loss, sampling, optimizer
│
├── evaluation/
│   ├── sample.py                # MoleculeSampler class
│   └── metrics.py               # Validity, uniqueness, diversity
│
├── checkpoints/
│   └── checkpoint_10000.pt      # Trained model (10K iterations)
│
└── demo/
    └── app.py                   # Streamlit UI
```

---

## 12. Questions for Deep Dive

When walking through with another Claude instance, consider exploring:

1. **State representation:** Why is build order part of the state space? How does this affect the GFlowNet DAG structure?

2. **Backward policy:** Why leaf removal instead of arbitrary fragment removal? What are the flow-matching implications?

3. **Action space factorization:** How does the factored policy avoid materializing the full `|APs| × V' × max_AP` space?

4. **Reward design:** Why log-space rewards? Why these specific weights (0.5, 0.3, 0.2)?

5. **Trajectory Balance vs alternatives:** What are the tradeoffs vs SubTB, Detailed Balance, Flow-Matching?

6. **Fragment vocabulary:** Why BRICS? What about RECAP or other decomposition schemes?

7. **Exploration:** Role of beta temperature? When to increase from 1.0 to 4-8?

8. **Mode collapse:** Why did the index bug cause complete collapse to single-fragment? What does this reveal about the gradient signal?

9. **Scalability:** Bottlenecks for larger vocab (500-1000 fragments)? Larger molecules (N_max=20)?

10. **Diversity vs reward:** Current uniqueness is 39%. Is this good? How to increase without sacrificing quality?

---

## 13. Next Steps (Beyond Current Implementation)

1. **Larger vocabulary:** Scale to 500-1000 fragments
2. **Protein binding:** Train docking surrogate model (requires ChEMBL/PDBbind data)
3. **Multi-objective:** Add ADMET properties (bioavailability, toxicity, clearance)
4. **Conditional generation:** Condition on target protein or property ranges
5. **Active learning:** Iterate with experimental validation

---

## References

**GFlowNets:**
- Bengio et al. "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation" (NeurIPS 2021)
- Malkin et al. "Trajectory Balance: Improved Credit Assignment in GFlowNets" (NeurIPS 2022)

**Fragment-based generation:**
- BRICS: Degen et al. "On the Art of Compiling and Using 'Drug-Like' Chemical Fragment Spaces" (ChemMedChem 2008)

**Molecular properties:**
- QED: Bickerton et al. "Quantifying the chemical beauty of drugs" (Nature Chemistry 2012)
- SA Score: Ertl & Schuffenhauer "Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions" (Journal of Cheminformatics 2009)

---

**This document provides complete ML architecture for FragFlow. Use it to ask detailed technical questions about any component.**
