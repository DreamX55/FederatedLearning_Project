"""
Privacy Accountant for Client-Level Differentially Private Federated Learning.
Implements exact Rényi Differential Privacy (RDP) accounting for Subsampled Gaussian Mechanisms
(Mironov et al., 2019; Wang et al., 2019).
"""

import math
import argparse
from typing import Tuple, Dict, List

def rdp_subsampled_gaussian_integer_order(alpha: int, sigma_global: float, q: float) -> float:
    """
    Computes RDP of order alpha for a subsampled Gaussian mechanism with sampling ratio q.
    Uses log-sum-exp formulation to ensure numerical stability.
    """
    if q == 1.0:
        return (alpha) / (2.0 * sigma_global ** 2)
    
    terms = []
    for j in range(alpha + 1):
        log_binom = math.lgamma(alpha + 1) - math.lgamma(j + 1) - math.lgamma(alpha - j + 1)
        log_q_term = (j * math.log(q) if j > 0 else 0.0) + ((alpha - j) * math.log(1.0 - q) if alpha - j > 0 else 0.0)
        exponent = (j * (j - 1.0)) / (2.0 * sigma_global ** 2)
        terms.append(log_binom + log_q_term + exponent)
    
    max_t = max(terms)
    sum_exp = sum(math.exp(t - max_t) for t in terms)
    log_total = max_t + math.log(sum_exp)
    
    rdp = (1.0 / (alpha - 1.0)) * log_total
    return max(0.0, rdp)

def compute_client_level_dp_epsilon(
    sigma_local: float,
    K: int = 10,
    N: int = 21,
    R: int = 50,
    delta: float = 1e-3,
    clip_norm: float = 1.0,
    max_alpha: int = 64
) -> Dict[str, float]:
    """
    Computes the (epsilon, delta) guarantee for client-level DP FedAvg.
    
    Args:
        sigma_local: Noise multiplier added to each client's clipped update delta.
        K: Number of clients sampled per round.
        N: Total number of federated clients in the training cohort.
        R: Number of communication rounds.
        delta: Target delta privacy parameter.
        clip_norm: Global L2 clipping norm bound C.
        max_alpha: Maximum Rényi order to evaluate.
        
    Returns:
        Dictionary containing epsilon, delta, optimal alpha, sigma_effective, and parameters.
    """
    if sigma_local <= 0.0:
        return {
            'epsilon': float('inf'),
            'delta': delta,
            'sigma_local': sigma_local,
            'sigma_global': 0.0,
            'q': K / N,
            'rounds': R,
            'optimal_alpha': 1,
            'clip_norm': clip_norm
        }
    
    q = K / N
    # Effective global noise multiplier relative to global sensitivity Delta_2 = C / K:
    # Client noise variance = sigma_local^2 * C^2 * I
    # Server noise on (1/K) sum delta_k has variance (sigma_local^2 * C^2 / K) * I
    # Sensitivity = C / K
    # Effective noise multiplier = (sigma_local * C / sqrt(K)) / (C / K) = sigma_local * sqrt(K)
    sigma_global = sigma_local * math.sqrt(K)
    
    alphas = list(range(2, max_alpha + 1))
    results = []
    
    for alpha in alphas:
        try:
            rdp_single_round = rdp_subsampled_gaussian_integer_order(alpha, sigma_global, q)
            rdp_total = R * rdp_single_round
            eps = rdp_total + math.log(1.0 / delta) / (alpha - 1.0)
            results.append((eps, alpha, rdp_total))
        except (OverflowError, ValueError):
            continue
            
    if not results:
        return {
            'epsilon': float('inf'),
            'delta': delta,
            'sigma_local': sigma_local,
            'sigma_global': sigma_global,
            'q': q,
            'rounds': R,
            'optimal_alpha': None,
            'clip_norm': clip_norm
        }
        
    min_eps, opt_alpha, total_rdp = min(results, key=lambda x: x[0])
    
    return {
        'epsilon': min_eps,
        'delta': delta,
        'sigma_local': sigma_local,
        'sigma_global': sigma_global,
        'q': q,
        'rounds': R,
        'optimal_alpha': opt_alpha,
        'total_rdp': total_rdp,
        'clip_norm': clip_norm
    }

def generate_privacy_tradeoff_table(
    sigmas: List[float] = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00],
    K: int = 10,
    N: int = 21,
    R: int = 50,
    delta: float = 1e-3
) -> List[Dict]:
    """Generates an evaluation table across multiple noise scales."""
    table = []
    for s in sigmas:
        res = compute_client_level_dp_epsilon(s, K=K, N=N, R=R, delta=delta)
        table.append(res)
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client-Level DP Privacy Accountant")
    parser.add_argument("--sigma", type=float, default=0.05, help="Local noise multiplier sigma")
    parser.add_argument("--K", type=int, default=10, help="Participating clients per round")
    parser.add_argument("--N", type=int, default=21, help="Total training clients")
    parser.add_argument("--R", type=int, default=50, help="Communication rounds")
    parser.add_argument("--delta", type=float, default=1e-3, help="Target delta parameter")
    parser.add_argument("--table", action="store_true", help="Print full noise multiplier sweep table")
    
    args = parser.parse_args()
    
    if args.table:
        print(f"\n{'='*80}")
        print(f"CLIENT-LEVEL RDP ACCOUNTING SWEEP (N={args.N}, K={args.K}, q={args.K/args.N:.4f}, R={args.R}, delta={args.delta})")
        print(f"{'='*80}")
        print(f"{'Local sigma':>12} | {'Global sigma':>12} | {'Epsilon (eps)':>15} | {'Delta':>10} | {'Opt Alpha':>10}")
        print(f"{'-'*80}")
        tbl = generate_privacy_tradeoff_table(K=args.K, N=args.N, R=args.R, delta=args.delta)
        for row in tbl:
            eps_str = f"{row['epsilon']:.4f}" if row['epsilon'] != float('inf') else "inf (No DP)"
            print(f"{row['sigma_local']:12.2f} | {row['sigma_global']:12.4f} | {eps_str:>15} | {row['delta']:10.1e} | {str(row['optimal_alpha']):>10}")
        print(f"{'='*80}\n")
    else:
        res = compute_client_level_dp_epsilon(args.sigma, K=args.K, N=args.N, R=args.R, delta=args.delta)
        print(f"Privacy Guarantee: (epsilon = {res['epsilon']:.4f}, delta = {res['delta']}) at sigma = {args.sigma}")
