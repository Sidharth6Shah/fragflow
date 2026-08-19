# FragFlow Examples

Practical examples for real-world drug discovery applications.

## Quick Start

### 1. Scaffold-Constrained Generation

Generate molecules containing a specific core structure (e.g., benzene ring, kinase hinge binder).

```bash
# Demo: See how scaffold constraints affect rewards
python examples/scaffold_demo.py

# Train: Generate benzene-containing molecules
python training/train_scaffold.py
```

**Use cases:**
- Fragment growing from known scaffold
- Scaffold hopping with structural constraints
- Generating analogs of existing drugs

**How it works:**
```python
# In training/config.py
SCAFFOLD_CONFIG = TrainingConfig(
    reward_mode="scaffold",
    scaffold_smarts="c1ccccc1",  # Benzene ring
    scaffold_weight=0.4,  # 40% weight on scaffold presence
    beta=4.0
)
```

Molecules containing the scaffold get a reward bonus:
- **With benzene**: Reward = 0.6 × (QED+SA+LogP) + 0.4 × 1.0
- **Without benzene**: Reward = 0.6 × (QED+SA+LogP) + 0.4 × 0.0

### 2. Target Protein Binding (Framework)

Optimize molecules for binding to a specific protein target.

```bash
# See framework structure
python reward/docking_reward.py
```

**Implementation steps:**

**Option A: Use Pre-trained Models (Fastest)**
```python
# Install ChemProp (pre-trained ADMET/binding models)
pip install chemprop

# Use pre-trained model
from chemprop import predict
affinity = predict(smiles, "models/egfr_binding.pt")
```

**Option B: Train Surrogate Model**
1. Download binding data from ChEMBL for your target
2. Train GNN to predict binding affinity
3. Use during GFlowNet training (fast inference)

**Option C: Direct Docking (Slow)**
```bash
# Install AutoDock Vina
conda install -c conda-forge vina

# Run docking for each molecule (10-60s per molecule)
# Not recommended during training - use for validation
```

### 3. Custom Reward Functions

Create your own reward based on any molecular property:

```python
# In reward/reward_fn.py

def my_custom_reward(mol, beta=1.0):
    """Optimize for your specific criteria."""

    # Example: Maximize molecular weight + LogP
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    # Normalize to [0, 1]
    mw_score = min(mw / 500, 1.0)
    logp_score = 1.0 / (1.0 + np.exp(-logp))

    R_raw = 0.5 * mw_score + 0.5 * logp_score
    return beta * np.log(R_raw + 1e-4)
```

## Example Scaffolds

Common SMARTS patterns for drug discovery:

```python
# Kinase hinge binders
"c1ncncc1"  # Pyrimidine
"c1cncnc1"  # Purine-like

# GPCR pharmacophores
"CCN(C)C"   # Basic amine
"c1ccc(O)cc1"  # Phenol

# Protease inhibitors
"NC(=O)C"   # Amide
"C(=O)O"    # Carboxylic acid

# Aromatic cores
"c1ccccc1"     # Benzene
"c1ccc2ccccc2c1"  # Naphthalene
"c1ccc2c(c1)ccc1ccccc12"  # Anthracene
```

## Protein Targets

Pre-defined targets with PDB IDs:

```python
PROTEIN_TARGETS = {
    "EGFR": "1M17",   # Cancer
    "BCL2": "4LVT",   # Cancer
    "CDK2": "1HCK",   # Cell cycle
    "HIV_PR": "3NU3", # HIV
    "ACE2": "6M0J",   # COVID-19
}
```

Download structures: https://www.rcsb.org/

## Training Tips

**For scaffold constraints:**
- Start with `scaffold_weight=0.3-0.5`
- Higher weight = stronger constraint (less diversity)
- Use broad SMARTS for flexibility

**For protein binding:**
- Combine with QED/SA to ensure drug-likeness
- Use `docking_weight=0.4-0.6`
- Train surrogate model on 10k+ compounds

**For multi-objective:**
```python
# Optimize binding + QED + specific MW range
R = 0.4*binding + 0.3*QED + 0.3*mw_score
```

## Next Steps

1. **Run scaffold demo**: `python examples/scaffold_demo.py`
2. **Try training**: `python training/train_scaffold.py`
3. **Evaluate results**: `python evaluation/metrics.py`
4. **Customize**: Edit `SCAFFOLD_CONFIG` with your scaffold

## Resources

- RDKit SMARTS: https://www.rdkit.org/docs/RDKit_Book.html
- ChEMBL data: https://www.ebi.ac.uk/chembl/
- AutoDock Vina: https://autodock-vina.readthedocs.io/
- PDB structures: https://www.rcsb.org/
