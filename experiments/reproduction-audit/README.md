# Reproduction Experiments: Respecting Modality Gap in Post-hoc OOD Detection

This directory contains numerical audits and experiments for reproducing the major claims of the paper "Respecting Modality Gap in Post-hoc Out-of-distribution Detection with Pre-trained Vision-Language Models" (arXiv:2605.26661).

## Experiments

### 1. Theorem 1: Optimal Visual Prototype in Visual Subspace
**File:** `audit_theorem1.py`

**Claim:** The optimal visual prototype w*_y lies entirely in the visual feature subspace S = span{z_1, ..., z_N}.

**Results:** SUPPORTED
- W* lies in span{z_i} to machine precision (~1e-15)
- Fixed-point iteration converges in ~10 iterations
- Self-consistency verified (max change < 1e-4)

### 2. Theorem 2: Modality Gap Lower Bound
**File:** `audit_theorem2.py`

**Claim:** ||R - W*||_F^2 >= sum 2(1 - ||Proj_S(r_y)||_2)

**Results:** SUPPORTED
- Inequality holds in all 5 trials
- Tight when text prototypes lie entirely in S (RHS → 0)
- Large margin when text prototypes have components outside S

### 3. Theorem 3: Cumulative Regret Bound O(√N)
**File:** `audit_theorem3.py`

**Claim:** Cumulative regret ≤ O(√N) with step size η_i = ρ/√i

**Results:** SUPPORTED
- Regret grows as C × √N (C ≈ 5.94)
- Coefficient of variation of regret/√N = 0.13 (< 0.3 threshold)
- Verified across N = 100, 200, 500, 1000, 2000

### 4. Pipeline Test
**File:** `test_minimal.py`

**Claim:** Algorithm 1 online pseudo-supervised framework works correctly

**Results:** PASS
- Pipeline runs end-to-end on synthetic data
- AUROC ~50% as expected with random features
- Streaming prototype adapter logic verified

## Running the Experiments

```bash
# Run all audits
python3 audit_theorem1.py
python3 audit_theorem2.py
python3 audit_theorem3.py
python3 test_minimal.py
```

## Requirements

- Python 3.8+
- NumPy
- scikit-learn (for AUROC computation)
- matplotlib (for Theorem 3 plot)

## Logbook

Full reproduction logbook published at:
https://huggingface.co/spaces/divaspoudel/repro-modality-gap-ood-detection
