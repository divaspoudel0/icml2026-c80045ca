"""
Numerical Audit of Theorem 2 (Modality Gap Lower Bound)

Theorem 2 states:
  ||R - W*||_F^2 >= sum_{y in Y} 2(1 - ||Proj_S(r_y)||_2)
"""

import numpy as np

np.random.seed(42)

def orthogonal_projector(S_basis):
    U, S, Vt = np.linalg.svd(S_basis.T, full_matrices=False)
    tol = max(S_basis.shape) * S[0] * 1e-10
    rank = np.sum(S > tol)
    Q = U[:, :rank]
    return Q @ Q.T

def project_onto_subspace(v, P_S):
    return P_S @ v

def compute_optimal_prototypes(z, y_labels, kappa=1.0):
    N, d = z.shape
    classes = np.unique(y_labels)
    K = len(classes)
    W = np.zeros((d, K))
    for k, c in enumerate(classes):
        mask = y_labels == c
        W[:, k] = z[mask].mean(axis=0)
        W[:, k] /= np.linalg.norm(W[:, k]) + 1e-8
    for iteration in range(500):
        W_old = W.copy()
        for k, c in enumerate(classes):
            mask = y_labels == c
            neg_mask = ~mask
            logits = z @ W / kappa
            logits -= logits.max(axis=1, keepdims=True)
            pi = np.exp(logits)
            pi = pi / pi.sum(axis=1, keepdims=True)
            pos_sum = ((1 - pi[mask, k])[:, None] * z[mask]).sum(axis=0)
            neg_sum = (pi[neg_mask, k][:, None] * z[neg_mask]).sum(axis=0)
            w_raw = pos_sum - neg_sum
            W[:, k] = w_raw / (np.linalg.norm(w_raw) + 1e-8)
        if np.max(np.abs(W - W_old)) < 1e-6:
            break
    return W

def numerical_audit_theorem2(d=64, N=30, K=10, n_text_dims_outside_S=10):
    print("=" * 70)
    print("NUMERICAL AUDIT OF THEOREM 2: Modality Gap Lower Bound")
    print("=" * 70)
    z_raw = np.random.randn(N, d)
    z = z_raw / np.linalg.norm(z_raw, axis=1, keepdims=True)
    y_labels = np.tile(np.arange(K), N // K)
    np.random.shuffle(y_labels)
    P_S = orthogonal_projector(z)
    P_S_perp = np.eye(d) - P_S
    W_star = compute_optimal_prototypes(z, y_labels, kappa=1.0)
    alpha = n_text_dims_outside_S / d
    R_in_S = P_S @ np.random.randn(d, K)
    R_in_S = R_in_S / np.linalg.norm(R_in_S, axis=0, keepdims=True)
    R_out_S = P_S_perp @ np.random.randn(d, K)
    R_out_S = R_out_S / (np.linalg.norm(R_out_S, axis=0, keepdims=True) + 1e-8)
    R = np.sqrt(1 - alpha) * R_in_S + np.sqrt(alpha) * R_out_S
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    lhs = np.linalg.norm(R - W_star, 'fro') ** 2
    rhs = 0.0
    for k in range(K):
        proj_norm = np.linalg.norm(project_onto_subspace(R[:, k], P_S))
        rhs += 2 * (1 - proj_norm)
    print(f"LHS: ||R - W*||_F^2 = {lhs:.6f}")
    print(f"RHS: sum 2(1 - ||Proj_S(r_y)||_2) = {rhs:.6f}")
    print(f"LHS >= RHS? {lhs >= rhs - 1e-8}  (margin: {lhs - rhs:.2e})")
    passed = lhs >= rhs - 1e-6
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return passed

if __name__ == "__main__":
    results = []
    for trial in range(5):
        np.random.seed(trial * 1000 + 42)
        results.append(numerical_audit_theorem2())
    print(f"\nMULTI-TRIAL: {sum(results)}/{len(results)} trials passed")
