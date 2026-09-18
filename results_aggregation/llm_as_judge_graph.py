import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# File mapping based on your uploaded CSVs
files = {
    "Llama 3B": {
        "base": "llama3b_task_level_results.csv",
        "ft": "llama3b_finetuned_task_level_results.csv"
    },
    "Llama 8B": {
        "base": "llama8b_task_level_results.csv",
        "ft": "llama8b_finetuned_task_level_results.csv" 
    },
    "Qwen 3B": {
        "base": "qwen3b_task_level_results.csv",
        "ft": "qwen3b_finetuned_task_level_results.csv" 
    }
}

# The new files contain different evaluation metrics
metrics = {
    'mean_score_1_to_10': 'Mean LLM-as-a-Judge Score (1 to 10)',
    'verdict_accuracy': 'Verdict Accuracy'
}

all_data = []

# Merge baseline and finetuned data to calculate the delta
for model_name, paths in files.items():
    df_base = pd.read_csv(paths["base"])
    df_ft = pd.read_csv(paths["ft"])
    df_merged = pd.merge(df_base, df_ft, on='task', suffixes=('_base', '_ft'))
    
    for metric_col in metrics.keys():
        delta_col = f"{metric_col}_delta"
        df_merged[delta_col] = df_merged[f"{metric_col}_ft"] - df_merged[f"{metric_col}_base"]
        
        for _, row in df_merged.iterrows():
            task_clean = row['task'].replace('_', ' ').title()
            all_data.append({
                "Task": task_clean,
                "Model": model_name,
                "Metric": metric_col,
                "Delta": row[delta_col]
            })

df_plot = pd.DataFrame(all_data)

# Set global seaborn parameters to match previous styling
sns.set_theme(style="white")
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
palette = {'Llama 3B': '#4c72b0', 'Llama 8B': '#55a868', 'Qwen 3B': '#c44e52'}

generated_images = []

for metric_key, metric_name in metrics.items():
    plt.figure(figsize=(12, 8))
    subset = df_plot[df_plot['Metric'] == metric_key]
    
    # Horizontal grouped bar chart
    ax = sns.barplot(
        data=subset, 
        x='Delta', 
        y='Task', 
        hue='Model',
        palette=palette
    )
    
    # Formatting layout
    plt.title(f"Fine-Tuning Impact by Task ({metric_name})", fontsize=16, pad=15)
    plt.xlabel("Score Change Delta", fontsize=13)
    plt.ylabel("")
    
    # Vertical zero-line and gridlines
    plt.axvline(0, color='black', linewidth=1.5)
    ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.4)
    ax.set_axisbelow(True)
    sns.despine(left=True)
    
    # Legend and labels
    plt.legend(title="", fontsize=12, loc='upper right', frameon=True)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save image
    filename = f"FineTuning_Impact_V2_{metric_key}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    generated_images.append(filename)
    plt.show()

# Zip images for download
zip_filename = "finetuning_impact_plots_v2.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for img in generated_images:
        zipf.write(img)