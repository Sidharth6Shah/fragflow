# Debug Scripts

Scripts used to diagnose bugs and test functionality during development.

## Files

- **test_multifrag.py** - Initial test that discovered multi-fragment assembly was broken
- **test_multifrag_v2.py** - Detailed test showing dummy atom indexing bug
- **test_fix.py** - Verification that the fix works (multi-fragment assembly now working)
- **validate_before_training.py** - Comprehensive pre-training validation (8 tests covering all critical components)

## Usage

Run from project root:
```bash
# Quick test: verify multi-fragment assembly works
python debug_scripts/test_fix.py

# Full validation: run before Lambda GPU training
python debug_scripts/validate_before_training.py
```

## Key Finding & Fix

These scripts revealed that multi-fragment molecule assembly was completely broken due to incorrect index mapping in `env/molecule_state.py:284`. Line counted removed atoms AFTER current index instead of BEFORE.

**Fixed by changing:** `if d > i` → `if d < i`

See DESIGN_DECISIONS.md #10 for full details. Bug is now FIXED.
