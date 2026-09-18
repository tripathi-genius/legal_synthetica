import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Map the base and finetuned CSV paths
files = {
    "Llama 3B": {
        "base": "llama_3b_task_level_results.csv",
        "ft": "llama3b_finetunedtask_level_results.csv"
    },
    "Llama 8B": {
        "base": "llama8btask_level_results.csv",
        "ft": "llama8b_finetuned_task_level_results.csv"
    },
    "Qwen 3B": {
        "base": "qwen_3b_task_level_results.csv",
        "ft": "qwen3b_finetuned_task_level_results.csv"
    }
}

metrics = {
    'avg_lexical_jaccard': 'Lexical Jaccard'
}

all_data = []

# Calculate the performance delta between finetuned and base models
for model_name, paths in files.items():
    df_base = pd.read_csv(paths["base"])
    df_ft = pd.read_csv(paths["ft"])
    df_merged = pd.merge(df_base, df_ft, on='task', suffixes=('_base', '_ft'))
    
    for metric_col in metrics.keys():
        delta_col = f"{metric_col}_delta"
        df_merged[delta_col] = df_merged[f"{metric_col}_ft"] - df_merged[f"{metric_col}_base"]
        
        for _, row in df_merged.iterrows():
            # Format task names for better readability
            task_clean = row['task'].replace('_', ' ').title()
            all_data.append({
                "Task": task_clean,
                "Model": model_name,
                "Metric": metric_col,
                "Delta": row[delta_col]
            })

df_plot = pd.DataFrame(all_data)

# Set global parameters to mimic the provided styling
sns.set_theme(style="white")
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

# Set custom palette for models
palette = {'Llama 3B': '#4c72b0', 'Llama 8B': '#55a868', 'Qwen 3B': '#c44e52'}

for metric_key, metric_name in metrics.items():
    plt.figure(figsize=(12, 8))
    subset = df_plot[df_plot['Metric'] == metric_key]
    
    # Generate the horizontal barplot
    ax = sns.barplot(
        data=subset, 
        x='Delta', 
        y='Task', 
        hue='Model',
        palette=palette
    )
    
    # Formatting to match the desired style
    plt.title(f"Fine-Tuning Impact by Task ({metric_name})", fontsize=16, pad=15)
    plt.xlabel("Score Change Between Finetuned vs Base Models", fontsize=13)
    plt.ylabel("")
    
    # Add vertical line strictly at 0
    plt.axvline(0, color='black', linewidth=1.5)
    
    # Grid formatting behind the bars
    ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.4)
    ax.set_axisbelow(True)
    
    # Cleanup visual clutter
    sns.despine(left=True)
    
    # Enhance labels
    plt.legend(title="", fontsize=12, loc='upper right', frameon=True)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('jaccard_score.png')
