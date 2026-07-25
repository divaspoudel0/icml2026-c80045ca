"""
Numerical Audit of Theorem 1 (Optimal Visual Prototype in Visual Subspace)

Theorem 1 states:
  w_y* = l2_normalize(sum_{(x_i, y_i) in D_y} (1 - pi_{iy}) z_i 
                     - sum_{(x_i, y_i) in D\D_y} pi_{iy} z_i)

The key structural claim is: w_y* lies in span{z_1, ..., z_N}.
"""

import numpy as np

np.random.seed(42)

def compute_optimal_prototypes_iterative(z, y_labels, kappa=1.0, max_iter=500, tol=1e-7):
    N, d = z.shape
    classes = np.unique(y_labels)
    K = len(classes)
    W = np.zeros((d, K))
    for k, c in enumerate(classes):
        mask = y_labels == c
        W[:, k] = z[mask].mean(axis=0)
        W[:, k] /= np.linalg.norm(W[:, k]) + 1e-8
    for iteration in range(max_iter):
        W_old = W.copy()
        logits = z @ W / kappa
        logits -= logits.max(axis=1, keepdims=True)
        pi = np.exp(logits)
        pi = pi / pi.sum(axis=1, keepdims=True)
        for k, c in enumerate(classes):
            mask = y_labels == c
            neg_mask = ~mask
            pos_sum = ((1 - pi[mask, k])[:, None] * z[mask]).sum(axis=0)
            neg_sum = (pi[neg_mask, k][:, None] * z[neg_mask]).sum(axis=0)
            w_raw = pos_sum - neg_sum
            W[:, k] = w_raw / (np.linalg.norm(w_raw) + 1e-8)
        if np.max(np.abs(W - W_old)) < tol:
            return W, iteration + 1
    return W, max_iter

def verify_in_span(W, z):
    d = z.shape[1]
    U, S, Vt = np.linalg.svd(z.T, full_matrices=False)
    tol = max(z.shape) * S[0] * 1e-10
    rank = np.sum(S > tol)
    Q = U[:, :rank]
    P_S = Q @ Q.T
    P_S_perp = np.eye(d) - P_S
    W_perp = P_S_perp @ W
    W_norms = np.linalg.norm(W, axis=0)
    perp_norms = np.linalg.norm(W_perp, axis=0)
    return perp_norms / (W_norms + 1e-8)

def verify_self_consistency(W, z, y_labels, kappa=1.0):
    N, d = z.shape
    classes = np.unique(y_labels)
    logits = z @ W / kappa
    logits -= logits.max(axis=1, keepdims=True)
    pi = np.exp(logits)
    pi = pi / pi.sum(axis=1, keepdims=True)
    max_change = 0.0
    for k, c in enumerate(classes):
        mask = y_labels == c
        neg_mask = ~mask
        pos_sum = ((1 - pi[mask, k])[:, None] * z[mask]).sum(axis=0)
        neg_sum = (pi[neg_mask, k][:, None] * z[neg_mask]).sum(axis=0)
        w_raw = pos_sum - neg_sum
        w_new = w_raw / (np.linalg.norm(w_raw) + 1e-8)
        change = np.linalg.norm(w_new - W[:, k])
        max_change = max(max_change, change)
    return max_change

def numerical_audit_theorem1(d=64, N=500, K=10, kappa=1.0):
    print("=" * 70)
    print("NUMERICAL AUDIT OF THEOREM 1: Optimal Visual Prototype in Span")
    print("=" * 70)
    print(f"Parameters: d={d}, N={N}, K={K}, kappa={kappa}")
    z_raw = np.random.randn(N, d)
    z = z_raw / np.linalg.norm(z_raw, axis=1, keepdims=True)
    y_labels = np.tile(np.arange(K), N // K)
    np.random.shuffle(y_labels)
    W_opt, iters = compute_optimal_prototypes_iterative(z, y_labels, kappa)
    print(f"Iterative optimization converged in {iters} iterations")
    rel_errors = verify_in_span(W_opt, z)
    print(f"\n--- Claim: W* lies in span{{z_1, ..., z_N}} ---")
    print(f"Max relative perpendicular error: {np.max(rel_errors):.2e}")
    print(f"Mean relative perpendicular error: {np.mean(rel_errors):.2e}")
    span_check = np.all(rel_errors < 1e-6)
    print(f"All < 1e-6? {span_check}")
    change = verify_self_consistency(W_opt, z, y_labels, kappa)
    print(f"\n--- Self-consistency check ---")
    print(f"Max change under update rule: {change:.2e}")
    converged = change < 1e-4
    print(f"Converged (< 1e-4)? {converged}")
    passed = span_check and converged
    print(f"\n{'=' * 70}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 70}")
    return passed

if __name__ == "__main__":
    results = []
    for trial in range(3):
        np.random.seed(trial * 1000 + 42)
        results.append(numerical_audit_theorem1())
    print(f"\nMULTI-TRIAL: {sum(results)}/{len(results)} trials passed")
