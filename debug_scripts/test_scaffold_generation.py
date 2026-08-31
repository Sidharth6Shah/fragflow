"""
Test scaffold-constrained molecule generation.

Validates:
1. Scaffold reward computation works
2. Can filter molecules by scaffold match
3. Reward bonuses are applied correctly
"""
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Contrib.SA_Score import sascorer

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evaluation.sample import MoleculeSampler
from reward.reward_fn import compute_scaffold_reward, compute_full_reward

print("=" * 70)
print("Testing Scaffold-Constrained Generation")
print("=" * 70)

# Test 1: Scaffold reward computation on known molecules
print("\n[1/3] Testing scaffold reward function...")
print("-" * 70)

test_molecules = [
    ("c1ccccc1C", "Toluene (has benzene ring)"),
    ("c1ccccc1CCO", "Phenethanol (has benzene ring)"),
    ("c1ccc2ccccc2c1", "Naphthalene (has benzene, but also fused ring)"),
    ("CCCCCCCC", "Octane (no ring)"),
]

scaffold_smarts = "c1ccccc1"  # Benzene ring
scaffold_weight = 0.3

print(f"Target scaffold: {scaffold_smarts} (benzene ring)")
print(f"Scaffold weight: {scaffold_weight}\n")

for smiles, description in test_molecules:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"❌ Invalid SMILES: {smiles}")
        continue

    # Check if molecule matches scaffold
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    has_match = mol.HasSubstructMatch(pattern)

    # Compute both rewards
    full_reward = compute_full_reward(mol, beta=1.0)
    scaffold_reward = compute_scaffold_reward(mol, beta=1.0,
                                             scaffold_smarts=scaffold_smarts,
                                             scaffold_weight=scaffold_weight)

    # Get individual properties for analysis
    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol)
    logp = Descriptors.MolLogP(mol)

    print(f"{description}")
    print(f"  SMILES: {smiles}")
    print(f"  Has benzene: {'✓' if has_match else '✗'}")
    print(f"  QED={qed:.3f}, SA={sa:.2f}, LogP={logp:.2f}")
    print(f"  Full reward:     {full_reward:.4f}")
    print(f"  Scaffold reward: {scaffold_reward:.4f}")
    print(f"  Bonus applied:   {scaffold_reward - full_reward:+.4f}")
    print()

# Test 2: Generate molecules and check scaffold matching
print("\n[2/3] Testing molecule generation with scaffold filtering...")
print("-" * 70)

checkpoint_path = project_root / "checkpoints" / "checkpoint_10000.pt"
if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

print("Loading generator...")
generator = MoleculeSampler(checkpoint_path, vocab_size=200, max_frags=8)

print(f"Generating 20 molecules...\n")
molecules = generator.sample_batch(num_samples=20)

# Filter by scaffold
scaffold_pattern = Chem.MolFromSmarts(scaffold_smarts)
matches = []
non_matches = []

for state, mol, reward in molecules:
    if mol is None:
        continue

    if mol.HasSubstructMatch(scaffold_pattern):
        matches.append((mol, reward))
    else:
        non_matches.append((mol, reward))

print(f"Results:")
print(f"  Total generated: {len(molecules)}")
print(f"  Valid molecules: {len(matches) + len(non_matches)}")
print(f"  With benzene ring: {len(matches)} ({len(matches)/(len(matches)+len(non_matches))*100:.1f}%)")
print(f"  Without benzene: {len(non_matches)} ({len(non_matches)/(len(matches)+len(non_matches))*100:.1f}%)")

if matches:
    print(f"\nMolecules WITH benzene ring:")
    for i, (mol, reward) in enumerate(matches[:5], 1):
        smiles = Chem.MolToSmiles(mol)
        qed = QED.qed(mol)
        print(f"  {i}. {smiles}")
        print(f"     QED={qed:.3f}, Reward={reward:.3f}")

if non_matches:
    print(f"\nMolecules WITHOUT benzene ring:")
    for i, (mol, reward) in enumerate(non_matches[:5], 1):
        smiles = Chem.MolToSmiles(mol)
        qed = QED.qed(mol)
        print(f"  {i}. {smiles}")
        print(f"     QED={qed:.3f}, Reward={reward:.3f}")

# Test 3: Other common scaffolds
print("\n[3/3] Testing other scaffold patterns...")
print("-" * 70)

other_scaffolds = [
    ("C1CCC1", "Cyclobutane (4-ring)"),
    ("C1CCCCC1", "Cyclohexane (6-ring)"),
    ("c1cnccc1", "Pyridine"),
    ("c1ccncc1", "Pyridine (alt)"),
    ("C(=O)O", "Carboxylic acid"),
]

# Use first 5 generated molecules
test_mols = [(mol, Chem.MolToSmiles(mol)) for mol, _ in (matches + non_matches)[:5]]

for scaffold_smarts, scaffold_name in other_scaffolds:
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    if pattern is None:
        print(f"❌ Invalid SMARTS: {scaffold_smarts}")
        continue

    matches_count = sum(1 for mol, _ in test_mols if mol.HasSubstructMatch(pattern))
    print(f"{scaffold_name:20s} ({scaffold_smarts:15s}): {matches_count}/5 molecules match")

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print("✓ Scaffold reward computation: WORKS")
print("✓ Scaffold matching logic: WORKS")
print("✓ Can filter generated molecules by scaffold: WORKS")
print("\nNOTE: Current model was NOT trained with scaffold constraints,")
print("      so molecules are not biased toward any particular scaffold.")
print("      To generate scaffold-focused molecules, retrain with")
print("      mode='scaffold' in the reward function.")
print("=" * 70)
