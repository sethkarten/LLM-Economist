#!/usr/bin/env python3
"""
Analyze bounded rationality experiment results.
Creates plots, tables, and updates LaTeX document.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

RESULTS_DIR = Path("results/bounded_100x2000")
OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MODELS = {
    "gemma3-4b": "Gemma-3-4B",
    "mistral-7b-v0.3": "Mistral-7B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "qwen3-8b": "Qwen3-8B",
    "olmo3-7b": "OLMo-3-7B"
}

SEEDS = [42, 123, 456]

def load_results():
    """Load all experiment results."""
    results = {}
    for model_id, model_name in MODELS.items():
        results[model_id] = {}
        for seed in SEEDS:
            filename = f"{model_id}_seed{seed}.json"
            filepath = RESULTS_DIR / filename
            if filepath.exists():
                with open(filepath) as f:
                    results[model_id][seed] = json.load(f)
                print(f"✓ Loaded {filename}")
            else:
                print(f"✗ Missing {filename}")
    return results

def plot_swf_convergence(results):
    """Plot SWF convergence for all models."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, len(MODELS)))

    for (model_id, model_name), color in zip(MODELS.items(), colors):
        if model_id not in results or not results[model_id]:
            continue

        # Collect all runs for this model
        all_swfs = []
        for seed in SEEDS:
            if seed in results[model_id]:
                metrics = results[model_id][seed].get('metrics_history', [])
                if len(metrics) >= 2000:
                    swf = [m['swf'] for m in metrics]
                    all_swfs.append(swf)

        if not all_swfs:
            continue

        # Compute mean and std
        all_swfs = np.array(all_swfs)
        mean_swf = all_swfs.mean(axis=0)
        std_swf = all_swfs.std(axis=0)
        steps = np.arange(len(mean_swf))

        # Plot mean with shaded std
        ax.plot(steps, mean_swf, label=model_name, color=color, linewidth=2)
        ax.fill_between(steps, mean_swf - std_swf, mean_swf + std_swf,
                        alpha=0.2, color=color)

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Social Welfare Function', fontsize=12)
    ax.set_title('SWF Convergence: Bounded Rationality (100 agents, 3 seeds)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'swf_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'swf_convergence.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved SWF convergence plot")
    plt.close()

def plot_tax_policies(results):
    """Plot final tax policies for each model."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    brackets = ['0-10k', '10-40k', '40-85k', '85-160k', '160-200k', '200-500k', '500k+']
    x_pos = np.arange(len(brackets))

    for idx, (model_id, model_name) in enumerate(MODELS.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        if model_id not in results or not results[model_id]:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(model_name)
            continue

        # Collect final tax rates
        all_rates = []
        for seed in SEEDS:
            if seed in results[model_id]:
                metrics = results[model_id][seed].get('metrics_history', [])
                if metrics:
                    final_rates = metrics[-1].get('tax_rates', [])
                    all_rates.append([r * 100 for r in final_rates])

        if all_rates:
            all_rates = np.array(all_rates)
            mean_rates = all_rates.mean(axis=0)
            std_rates = all_rates.std(axis=0)

            ax.bar(x_pos, mean_rates, yerr=std_rates, capsize=5,
                  alpha=0.7, color='steelblue')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(brackets, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Tax Rate (%)', fontsize=10)
            ax.set_title(model_name, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 100)

    # Hide unused subplot
    if len(MODELS) < len(axes):
        axes[-1].axis('off')

    plt.suptitle('Final Tax Policies by Model', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tax_policies.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'tax_policies.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved tax policies plot")
    plt.close()

def generate_summary_table(results):
    """Generate summary statistics table."""
    table_rows = []

    for model_id, model_name in MODELS.items():
        if model_id not in results or not results[model_id]:
            continue

        swf_finals = []
        swf_means = []
        swf_maxs = []

        for seed in SEEDS:
            if seed in results[model_id]:
                metrics = results[model_id][seed].get('metrics_history', [])
                if len(metrics) >= 2000:
                    swfs = [m['swf'] for m in metrics]
                    swf_finals.append(swfs[-1])
                    swf_means.append(np.mean(swfs))
                    swf_maxs.append(np.max(swfs))

        if swf_finals:
            row = {
                'Model': model_name,
                'Final SWF': f"{np.mean(swf_finals):.1f} ± {np.std(swf_finals):.1f}",
                'Mean SWF': f"{np.mean(swf_means):.1f} ± {np.std(swf_means):.1f}",
                'Max SWF': f"{np.mean(swf_maxs):.1f} ± {np.std(swf_maxs):.1f}",
                'Seeds': len(swf_finals)
            }
            table_rows.append(row)

    # Print LaTeX table
    print("\n" + "="*80)
    print("LaTeX Table:")
    print("="*80)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Bounded Rationality Results: 100 Agents, 2000 Timesteps}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Model & Final SWF & Mean SWF & Max SWF & Seeds \\")
    print(r"\midrule")
    for row in table_rows:
        print(f"{row['Model']} & {row['Final SWF']} & {row['Mean SWF']} & {row['Max SWF']} & {row['Seeds']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:bounded_results}")
    print(r"\end{table}")
    print("="*80 + "\n")

    # Save to file
    with open(OUTPUT_DIR / 'summary_table.tex', 'w') as f:
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Bounded Rationality Results: 100 Agents, 2000 Timesteps}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Model & Final SWF & Mean SWF & Max SWF & Seeds \\" + "\n")
        f.write(r"\midrule" + "\n")
        for row in table_rows:
            f.write(f"{row['Model']} & {row['Final SWF']} & {row['Mean SWF']} & {row['Max SWF']} & {row['Seeds']} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\label{tab:bounded_results}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"✓ Saved LaTeX table to {OUTPUT_DIR / 'summary_table.tex'}")

def main():
    print("="*80)
    print("BOUNDED RATIONALITY EXPERIMENT ANALYSIS")
    print("="*80 + "\n")

    print("Loading results...")
    results = load_results()

    print("\nGenerating plots...")
    plot_swf_convergence(results)
    plot_tax_policies(results)

    print("\nGenerating summary table...")
    generate_summary_table(results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - swf_convergence.pdf/png")
    print("  - tax_policies.pdf/png")
    print("  - summary_table.tex")
    print("\nNext steps:")
    print("  1. Review plots in results/analysis/")
    print("  2. Copy summary_table.tex into LaTeX document")
    print("  3. Start RL AI Economist baseline training")

if __name__ == '__main__':
    main()
