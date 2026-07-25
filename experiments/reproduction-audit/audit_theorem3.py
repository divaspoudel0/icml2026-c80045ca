"""
Numerical Audit of Theorem 3 (Cumulative Regret Bound O(sqrt(N)))

Theorem 3 states:
  sum_{i=1}^N ell_hat(W^{(i-1)}; x_i, Y_hat) - min_W L_hat(W) <= O(sqrt(N))
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

def compute_ell_hat(W, x, text_protos, visual_temperature=0.05):
    vis_logits = x @ W / visual_temperature
    vis_logits -= vis_logits.max()
    vis_probs = np.exp(vis_logits)
    vis_probs = vis_probs / (vis_probs.sum() + 1e-8)
    text_logits = x @ text_protos / 0.01
    text_logits -= text_logits.max()
    text_probs = np.exp(text_logits)
    text_probs = text_probs / (text_probs.sum() + 1e-8)
    eps = 1e-8
    loss = -np.sum(text_probs * np.log(vis_probs + eps))
    return loss

def compute_gradient(W, x, text_protos, visual_temperature=0.05):
    vis_logits = x @ W / visual_temperature
    vis_logits -= vis_logits.max()
    vis_probs = np.exp(vis_logits)
    vis_probs = vis_probs / (vis_probs.sum() + 1e-8)
    text_logits = x @ text_protos / 0.01
    text_logits -= text_logits.max()
    text_probs = np.exp(text_logits)
    text_probs = text_probs / (text_probs.sum() + 1e-8)
    grad = np.outer(x, (vis_probs - text_probs)) / visual_temperature
    return grad

def online_descent_regret(d=32, K=5, N=1000, rho=0.1, visual_temperature=0.05):
    R = np.random.randn(d, K)
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    W = R.copy()
    X = np.random.randn(N, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    W_offline = R.copy()
    for _ in range(100):
        total_grad = np.zeros_like(W_offline)
        for i in range(N):
            total_grad += compute_gradient(W_offline, X[i], R, visual_temperature)
        W_offline -= 0.01 * total_grad / N
        W_offline = W_offline / np.linalg.norm(W_offline, axis=0, keepdims=True)
    min_loss = np.mean([compute_ell_hat(W_offline, X[i], R, visual_temperature) 
                        for i in range(min(N, 500))])
    cumulative_regret = 0.0
    regrets = []
    for i in range(N):
        current_loss = compute_ell_hat(W, X[i], R, visual_temperature)
        step_regret = current_loss - min_loss
        cumulative_regret += step_regret
        regrets.append(cumulative_regret)
        grad = compute_gradient(W, X[i], R, visual_temperature)
        eta = rho / np.sqrt(i + 1)
        W = W - eta * grad
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
    return regrets, cumulative_regret

def numerical_audit_theorem3():
    print("=" * 70)
    print("NUMERICAL AUDIT OF THEOREM 3: Cumulative Regret Bound O(sqrt(N))")
    print("=" * 70)
    N_values = [100, 200, 500, 1000, 2000]
    final_regrets = []
    for N in N_values:
        np.random.seed(42)
        regrets, final_regret = online_descent_regret(d=32, K=5, N=N, rho=0.1)
        final_regrets.append(final_regret)
        print(f"N={N:5d}: cumulative regret = {final_regret:.4f}")
    N_arr = np.array(N_values, dtype=float)
    regret_arr = np.array(final_regrets)
    C_fit = np.sum(regret_arr * np.sqrt(N_arr)) / np.sum(N_arr)
    print(f"\nEstimated C: {C_fit:.4f}")
    ratios = [r / np.sqrt(N) for N, r in zip(N_values, final_regrets)]
    cv = np.std(ratios) / np.mean(ratios)
    print(f"Coefficient of variation of regret/sqrt(N): {cv:.4f}")
    print(f"Regret grows as O(sqrt(N))? CV < 0.3 -> {'YES' if cv < 0.3 else 'NO'}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(N_values, final_regrets, 'bo-', label='Actual regret')
    axes[0].plot(N_values, C_fit * np.sqrt(N_values), 'r--', label=f'{C_fit:.3f} * sqrt(N)')
    axes[0].set_xlabel('N (number of samples)')
    axes[0].set_ylabel('Cumulative Regret')
    axes[0].set_title('Theorem 3: Regret Bound')
    axes[0].legend()
    axes[0].grid(True)
    np.random.seed(42)
    regrets_1000, _ = online_descent_regret(d=32, K=5, N=1000, rho=0.1)
    axes[1].plot(range(1, 1001), regrets_1000, 'b-', alpha=0.7, label='Cumulative regret')
    axes[1].plot(range(1, 1001), C_fit * np.sqrt(range(1, 1001)), 'r--', label=f'{C_fit:.3f} * sqrt(N)')
    axes[1].set_xlabel('Step i')
    axes[1].set_ylabel('Cumulative Regret')
    axes[1].set_title('Regret Trajectory (N=1000)')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('/tmp/theorem3_regret.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to /tmp/theorem3_regret.png")
    passed = cv < 0.3
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return passed

if __name__ == "__main__":
    numerical_audit_theorem3()
