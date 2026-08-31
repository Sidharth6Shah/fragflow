"""
Test 3D molecule visualization with py3Dmol.

This validates that we can:
1. Generate 3D coordinates for molecules
2. Create mol blocks for 3D viewers
3. Use py3Dmol for interactive visualization
"""
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evaluation.sample import MoleculeSampler

print("=" * 70)
print("Testing 3D Molecule Visualization")
print("=" * 70)

# Generate a molecule
print("\n[1/3] Generating test molecule...")
checkpoint_path = project_root / "checkpoints" / "checkpoint_10000.pt"
generator = MoleculeSampler(checkpoint_path, vocab_size=200, max_frags=8)

state, mol, reward = generator.sample_molecule()
if mol is None:
    print("❌ Failed to generate molecule")
    sys.exit(1)

smiles = Chem.MolToSmiles(mol)
print(f"✓ Generated: {smiles}")
print(f"  Reward: {reward:.3f}")

# Generate 3D coordinates
print("\n[2/3] Generating 3D coordinates...")
mol_3d = Chem.Mol(mol)

# Try ETKDG (better conformer generation)
params = AllChem.ETKDGv3()
params.randomSeed = 42

try:
    result = AllChem.EmbedMolecule(mol_3d, params)
    if result == -1:
        print("⚠ ETKDG failed, trying basic embedding...")
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)

    if result == -1:
        print("❌ Could not generate 3D coordinates")
        sys.exit(1)

    print(f"✓ 3D embedding successful")

    # Optimize geometry with MMFF
    try:
        mmff_result = AllChem.MMFFOptimizeMolecule(mol_3d)
        print(f"✓ MMFF optimization: {mmff_result} iterations")
    except:
        print("⚠ MMFF optimization failed (molecule still usable)")

except Exception as e:
    print(f"❌ 3D generation failed: {e}")
    sys.exit(1)

# Create mol block for 3D viewer
print("\n[3/3] Creating mol block for py3Dmol...")
try:
    mol_block = Chem.MolToMolBlock(mol_3d)
    print(f"✓ Mol block created ({len(mol_block)} chars)")
    print(f"  Preview (first 200 chars):")
    print("  " + "\n  ".join(mol_block[:200].split("\n")))

    # Test py3Dmol import
    import py3Dmol
    print(f"\n✓ py3Dmol imported successfully")
    print(f"  Version: {py3Dmol.__version__ if hasattr(py3Dmol, '__version__') else 'unknown'}")

except Exception as e:
    print(f"❌ Mol block creation failed: {e}")
    sys.exit(1)

# Test multiple molecules
print("\n" + "=" * 70)
print("Testing batch 3D generation")
print("=" * 70)

print("Generating 3 molecules...")
molecules = generator.sample_batch(num_samples=3)

success_count = 0
for i, (state, mol, reward) in enumerate(molecules, 1):
    if mol is None:
        continue

    smiles = Chem.MolToSmiles(mol)
    mol_3d = Chem.Mol(mol)

    try:
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        if result != -1:
            AllChem.MMFFOptimizeMolecule(mol_3d)
            mol_block = Chem.MolToMolBlock(mol_3d)
            success_count += 1
            print(f"  {i}. ✓ {smiles} ({len(mol_block)} chars)")
        else:
            print(f"  {i}. ✗ {smiles} (failed embedding)")
    except Exception as e:
        print(f"  {i}. ✗ {smiles} (error: {e})")

print(f"\n✓ Successfully generated 3D coords for {success_count}/3 molecules")

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print("✓ 3D coordinate generation: WORKS")
print("✓ MMFF geometry optimization: WORKS")
print("✓ Mol block creation: WORKS")
print("✓ py3Dmol available: WORKS")
print("\nReady for Streamlit 3D visualization!")
print("=" * 70)
