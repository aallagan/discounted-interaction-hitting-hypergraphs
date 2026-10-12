# Discounted Interaction-Hitting Domination on Hypergraphs

Reproducibility code for the paper:

Julian D. Allagan, Kevin Pereyra, and William A. Massey,
"Discounted Interaction-Hitting Domination on Hypergraphs."

## Files

- `verify_recovery.py` — checks the sharp recovery windows, extremal
  constructions, staircase theorem, cubic specialization, codegree bound,
  and projective-plane separation.

- `verify_loose.py` — checks the loose-path and loose-cycle formulas,
  transfer-matrix counts, feasibility criteria, and fault-tolerance results.

## Requirements

Python 3 with:

- numpy
- scipy
- networkx

Install with:

    pip install -r requirements.txt

## Running the verification

    python verify_recovery.py
    python verify_loose.py

A successful run ends with:

    All checks passed.

## Reproducibility

Equilibrium systems and the principal finite-instance checks use exact
rational arithmetic. Some analytic closed-form comparisons use floating
point evaluation, and the larger covering instances are solved using
`scipy.optimize.milp`.

Randomized tests use fixed random seeds for reproducibility.

## Notes

The projective-plane implementation in `verify_recovery.py` is used for
q = 2 and q = 3. The simple modular implementation should not be interpreted
as a general finite-field implementation for arbitrary prime powers.

## Paper

[Paper link will be added when publicly available.]
