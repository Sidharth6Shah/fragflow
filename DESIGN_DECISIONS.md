# FragFlow Design Decisions

Quick reference for all fixes and design choices made during development.

---

## 1. FIXED MOLECULE ASSEMBLY (`env/molecule_state.py:170-298`)

**Problem:**
Original `state_to_molecule()` just combined fragments side-by-side without connecting them, resulting in disconnected pieces like `[*]c1ccccc1.[*]CCCC`. Generated "molecules" weren't real compounds. Evaluation showed "100% validity" but was meaningless.

**Solution:**
Implemented proper BRICS reconstruction: track dummy atom positions, map bonds from state to atom indices, connect real atoms by bonding neighbors of dummies, remove all dummy atoms with index remapping. Now generates real molecules like `CCCc1ccccc1-c1ccc(C)cc1`.

---

## 2. ADDED GRADIENT CLIPPING (`training/train.py:214-219`)

**Problem:**
Training diverged at iteration 15300 (loss: 0.03 → 210, reward: -0.38 → -1.2). Squared TB loss `(log_Z + log_pf - log_pb - log_R)^2` amplifies large values exponentially, causing gradient explosion. 40 hours of training wasted.

**Solution:**
Added `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` before optimizer step. Limits gradient magnitude to prevent explosion while allowing training. Standard practice in RL/GFlowNet literature.

---

## 3. INCREASED BETA 1.0 → 4.0 (`training/config.py:61`)

**Problem:**
With beta=1.0, model stopped early (95.7% molecules had only 2 fragments), low uniqueness (37%), poor exploration. Beta controls reward temperature: `logR = beta * log(R_raw)`, so low beta doesn't differentiate between rewards enough.

**Solution:**
Increased beta to 4.0 in FULL_CONFIG. Amplifies reward differences to encourage exploration and longer molecules. Typical range for molecule generation: 2-8 (from literature).

---

## 4. SCAFFOLD-CONSTRAINED GENERATION (`reward/reward_fn.py:92-137`)

**Problem:**
Original system generated random molecules with no structural constraints. Real drug discovery needs fragment growing from known scaffolds, scaffold hopping, and analog generation with specific cores.

**Solution:**
Added scaffold reward mode: check if molecule contains SMARTS pattern using `mol.HasSubstructMatch(scaffold_mol)`, give bonus if present, combine with base reward via `R = (1-w)*base_reward + w*scaffold_bonus`. Enables targeted molecule design while maintaining drug-likeness.

---

## 5. PROTEIN BINDING FRAMEWORK (`reward/docking_reward.py`)

**Problem:**
Generic drug-likeness (QED/SA/LogP) doesn't optimize for specific therapeutic targets. Need binding affinity prediction for proteins like EGFR, BCL2, etc.

**Solution:**
Created framework with 3 implementation options: (A) pre-trained ChemProp models (fast), (B) train GNN surrogate on ChEMBL data (accurate), (C) direct AutoDock Vina (slow, gold standard). Provided structure and 5 common protein targets (EGFR, BCL2, CDK2, HIV_PR, ACE2).

---

## 6. TRAINING CONFIG ORGANIZATION (`training/config.py`)

**Problem:**
Single config made it hard to switch experiments, compare reward functions, or reproduce runs. Different use cases (testing, full training, scaffold constraints) need different hyperparameters.

**Solution:**
Created named configs: MILESTONE_CONFIG (tiny vocab, dummy reward for validation), FULL_CONFIG (full vocab, real reward for generic drugs), SCAFFOLD_CONFIG (benzene constraint for targeted design). Easy to run experiments with clear hyperparameter documentation.

---

## 7. EVALUATION METRICS FIX (`evaluation/metrics.py:19-89`)

**Problem:**
Original evaluation assumed SMILES strings, but we changed sampling to return RDKit Mol objects to avoid lossy re-parsing and preserve molecular properties.

**Solution:**
Updated all metric functions to handle Mol objects: convert to SMILES only when needed for deduplication (`smiles = Chem.MolToSmiles(mol)`), work with Mol objects directly for property computation. More efficient and preserves data.

---

## 8. INDEX MAPPING AFTER DUMMY REMOVAL (`env/molecule_state.py:275-290`)

**Problem:**
Removing dummy atoms shifts indices: `[C(0), C(1), *(2), C(3), *(4), C(5)]` becomes `[C(0), C(1), C(2), C(3)]` after removing `*(2)` and `*(4)`. Trying to bond to old index 5 fails.

**Solution:**
Build index mapping before removal: count removed atoms before each index, subtract to get new index (`new_idx = old_idx - removed_before`). Use mapped indices when creating bonds after dummy removal.

---

## 9. PARAMS VARIABLE FOR GRADIENT CLIPPING (`training/train.py:69-76, 217`)

**Problem:**
Gradient clipping needs all parameters, but they're scattered across models: frag_embed, encoder, forward_policy, backward_policy, log_Z. Forgetting one causes incomplete clipping.

**Solution:**
Store all params in single list at initialization: `params = list(frag_embed.parameters()) + ... + [log_Z]`. Use for both optimizer and clipping. Single source of truth prevents bugs.

---

## Metrics Summary

**Before fixes:** Validity 100% (disconnected), Uniqueness 37%, Diversity 73%, Training diverged at 15300

**After fixes:** Validity 83.5% (connected), Uniqueness 43%, Diversity 79.6%, Training stable

**New features:** Scaffold constraints, protein binding framework, beta=4.0 exploration
