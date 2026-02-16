#!/usr/bin/env python3
"""
Visualize the non-stationary SWF landscape.

Shows how the optimal tax policy shifts as worker behavior evolves,
motivating the need for adaptive mechanism design (ICRL/REINFORCE++).

Worker model (GHH preferences):
  - Skills drawn from log-normal distribution
  - Utility: u_i = log(c_i - psi * l_i^(1+1/eps_i) / (1+1/eps_i))
    where c_i is consumption (post-tax income + rebate)
  - This is Greenwood-Hercowitz-Huffman (GHH) utility, where labor
    and consumption are non-separable. In GHH, higher taxes reduce
    labor AND utility through the log term.
  - Heterogeneous Frisch elasticities eps_i drift over time.

SWF = mean utility = (1/N) sum_i u_i

Output: 2x2 grid of contour plots at t=0, 500, 1000, 2000
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BRACKET_THRESHOLD = 80_000


def generate_population(n=200, seed=42):
    """Generate worker population."""
    np.random.seed(seed)
    skills = np.exp(np.random.normal(7.3, 0.6, n))
    skill_rank = np.argsort(np.argsort(skills)).astype(float) / n

    base_eps = 0.35 + 0.10 * np.random.randn(n)
    base_eps = np.clip(base_eps, 0.15, 0.60)

    drift = np.zeros(n)
    top = skill_rank > 0.5
    bot = skill_rank <= 0.5
    drift[top] = np.random.uniform(0.00006, 0.00014, top.sum())
    drift[bot] = np.random.uniform(-0.00008, -0.00002, bot.sum())

    return skills, base_eps, drift


def optimal_labor(skill, low_rate, high_rate, eps, psi=0.5, l_max=80.0):
    """Optimal labor for GHH preferences.

    u = log(c - psi * l^(1+1/eps) / (1+1/eps))
    c = skill * l * (1 - marginal_rate) [approximate for FOC]

    Grid search over l.
    """
    L = np.linspace(0.5, l_max, 200)
    gross = skill * L

    # Two-bracket tax
    tax = np.minimum(gross, BRACKET_THRESHOLD) * low_rate \
        + np.maximum(gross - BRACKET_THRESHOLD, 0.0) * high_rate
    c = np.maximum(gross - tax, 0.01)

    # GHH: consumption net of labor disutility
    exp = 1.0 + 1.0 / max(eps, 0.08)
    labor_cost = psi * L ** exp / exp
    arg = c - labor_cost
    arg = np.maximum(arg, 1e-6)

    u = np.log(arg)
    return L[np.argmax(u)]


def compute_swf(skills, elasticity, low_rate, high_rate, psi=0.5):
    """Compute SWF = mean(u_i) with GHH utility and redistribution."""
    n = len(skills)

    labor = np.array([
        optimal_labor(skills[i], low_rate, high_rate, elasticity[i], psi)
        for i in range(n)
    ])

    gross = skills * labor
    tax = np.minimum(gross, BRACKET_THRESHOLD) * low_rate \
        + np.maximum(gross - BRACKET_THRESHOLD, 0.0) * high_rate
    rebate = tax.sum() / n
    c = np.maximum(gross - tax + rebate, 0.01)

    exp = 1.0 + 1.0 / np.maximum(elasticity, 0.08)
    labor_cost = psi * labor ** exp / exp
    arg = np.maximum(c - labor_cost, 1e-6)

    return float(np.mean(np.log(arg)))


def sweep_landscape(skills, elasticity, low_rates, high_rates, psi=0.5):
    nr, nc = len(high_rates), len(low_rates)
    grid = np.zeros((nr, nc))
    total = nr * nc
    count = 0
    for i, hr in enumerate(high_rates):
        for j, lr in enumerate(low_rates):
            grid[i, j] = compute_swf(skills, elasticity, lr, hr, psi)
            count += 1
            if count % 100 == 0:
                print(f"  {count}/{total}", flush=True)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-agents', type=int, default=200)
    parser.add_argument('--grid-size', type=int, default=30)
    parser.add_argument('--output', type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent
                                    / 'llm-economist-latex' / 'fig'
                                    / 'swf_landscape.pdf'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.num_agents} workers ...")
    skills, base_eps, drift = generate_population(args.num_agents, args.seed)
    inc40 = skills * 40
    print(f"  Income@40h: [{inc40.min():.0f}, {inc40.max():.0f}]")
    print(f"  Frac above {BRACKET_THRESHOLD:,}: {(inc40 > BRACKET_THRESHOLD).mean():.1%}")

    low_rates = np.linspace(0.02, 0.80, args.grid_size)
    high_rates = np.linspace(0.02, 0.80, args.grid_size)

    timesteps = [0, 500, 1000, 2000]
    titles = ['$t = 0$', '$t = 500$', '$t = 1{,}000$', '$t = 2{,}000$']

    grids, optima = [], []

    for t in timesteps:
        print(f"\nComputing t={t} ...")
        eps = np.clip(base_eps + drift * t, 0.08, 2.0)
        print(f"  eps: [{eps.min():.3f}, {eps.max():.3f}]")

        grid = sweep_landscape(skills, eps, low_rates, high_rates)
        grids.append(grid)

        opt_idx = np.unravel_index(grid.argmax(), grid.shape)
        oh, ol = high_rates[opt_idx[0]], low_rates[opt_idx[1]]
        optima.append((ol, oh, grid.max()))
        print(f"  Opt: low={ol*100:.1f}%, high={oh*100:.1f}%, SWF={grid.max():.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2))
    lr_pct, hr_pct = low_rates * 100, high_rates * 100

    for idx, (grid, title, (ol, oh, _)) in enumerate(zip(grids, titles, optima)):
        ax = axes.flat[idx]
        vlo, vhi = np.percentile(grid, 2), np.percentile(grid, 98)
        levels = np.linspace(vlo, vhi, 25)

        im = ax.contourf(lr_pct, hr_pct, grid, levels=levels,
                         cmap='RdYlBu_r', extend='both')
        ax.contour(lr_pct, hr_pct, grid, levels=levels,
                   colors='k', linewidths=0.3, alpha=0.25)

        ox, oy = ol * 100, oh * 100
        ax.plot(ox, oy, 'k*', markersize=16,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)

        tx = ox + (-14 if ox > 40 else 4)
        ty = oy + (-6 if oy > 40 else 4)
        ax.annotate(f'({ox:.0f}%, {oy:.0f}%)',
                    xy=(ox, oy), xytext=(tx, ty), fontsize=8.5,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='gray', alpha=0.85),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Low-bracket rate (%)', fontsize=11)
        ax.set_ylabel('High-bracket rate (%)', fontsize=11)
        ax.tick_params(labelsize=10)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label('SWF', fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle('Social Welfare Landscape: Non-Stationarity in Tax Optimization',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved -> {output_path}")
    print(f"Saved -> {output_path.with_suffix('.png')}")

    print("\n" + "=" * 60)
    for t, (ol, oh, swf) in zip(timesteps, optima):
        print(f"  t={t:>5}: low={ol*100:5.1f}%, high={oh*100:5.1f}%, SWF={swf:.4f}")
    print("=" * 60)

    opt_lows = [o[0]*100 for o in optima]
    opt_highs = [o[1]*100 for o in optima]
    ls = max(opt_lows) - min(opt_lows)
    hs = max(opt_highs) - min(opt_highs)
    if ls < 3 and hs < 3:
        print("WARNING: Optima barely moved.")
    else:
        print(f"Shift: low {ls:.1f}pp, high {hs:.1f}pp")


if __name__ == '__main__':
    main()
