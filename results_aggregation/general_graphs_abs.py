import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
files = [
    "legalbench_reasoning_results_llama3b_base.csv",
    "legalbench_reasoning_results_llama3b_finetuned.csv",
    "legalbench_reasoning_results_llama8b_base.csv",
    "legalbench_reasoning_results_llama8b_finetuned.csv",
    "legalbench_reasoning_results_qwen3b_base.csv",
    "legalbench_reasoning_results_qwen3b_finetuned.csv"
]
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs)

# Pivot to compare Base vs Finetuned side-by-side per model
pivot = combined.pivot(index='task', columns='model', values='score')
if 'successor_liability' in pivot.index:
    pivot = pivot.drop('successor_liability')

tasks = pivot.index.tolist()
models = [('Llama 3B', 'llama-3b-base', 'llama-3b-finetuned'),
          ('Llama 8B', 'llama-8b-base', 'llama-8b-finetuned'),
          ('Qwen 3B', 'qwen-3b-base', 'qwen-3b-finetuned')]

model_labels = [m[0] for m in models]
base_cols = [m[1] for m in models]
ft_cols = [m[2] for m in models]

# Set up the figure with a 4x2 grid for the 8 valid tasks
fig, axes = plt.subplots(4, 2, figsize=(15, 18))
axes = axes.flatten()

x = np.arange(len(model_labels))
width = 0.35

for i, task in enumerate(tasks):
    ax = axes[i]
    
    # Extract data for the current task
    base_scores = pivot.loc[task, base_cols].values
    ft_scores = pivot.loc[task, ft_cols].values
    deltas = ft_scores - base_scores
    
    # Plot the grouped bars for Base and Fine-Tuned
    bars1 = ax.bar(x - width/2, base_scores, width, label='Base', color='#4C72B0', alpha=0.8)
    bars2 = ax.bar(x + width/2, ft_scores, width, label='Fine-Tuned', color='#55A868', alpha=0.8)
    
    # Customize the primary axis (absolute scores)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(task.replace('_', ' ').title(), fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    
    # Create a secondary y-axis for the delta line
    ax2 = ax.twinx()
    # Plot the delta line
    line = ax2.plot(x, deltas, color='#C44E52', marker='o', linewidth=2, markersize=8, label='Delta (FT - Base)')
    
    # Customize the secondary axis (deltas)
    ax2.set_ylabel('Delta', fontsize=10, color='#C44E52')
    ax2.tick_params(axis='y', labelcolor='#C44E52')
    
    # Center the 0-line for the delta axis to make positive/negative changes instantly recognizable
    max_abs_delta = max(abs(deltas.min()), abs(deltas.max())) + 0.05
    ax2.set_ylim(-max_abs_delta, max_abs_delta)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add legends (only to the very first subplot to reduce visual clutter)
    if i == 0:
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(0.0, 1.25), ncol=3)

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(top=0.92) # Leave room for the main title and legend
plt.suptitle('Model Performance and Fine-Tuning Delta per Task', fontsize=16, fontweight='bold', y=0.98)

plt.savefig('task_grouped_bar_delta.png', dpi=300, bbox_inches='tight')
plt.show()
