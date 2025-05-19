import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Enforce the use of non interactive backend to prevent segment errors
plt.switch_backend('Agg')

project_dir = "/your_path/EnDeep4mC/evaluations/ablation_model_indiv"
output_dir = os.path.join(project_dir, "viz")
os.makedirs(output_dir, exist_ok=True)

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'axes.titleweight': 'semibold',
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'font.weight': 'semibold',
    'axes.labelweight': 'semibold',
    'font.family': 'DejaVu Sans',
    'figure.titlesize': 20,
    'legend.title_fontsize': 20,
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.5,
    'xtick.major.size': 4,
    'ytick.major.size': 3,
    'xtick.major.width': 1,
    'ytick.major.width': 1
})

def load_and_process_data():
    all_dfs = []
    for file in os.listdir(project_dir):
        if file.endswith("_results.csv"):
            try:
                base_name = file.replace('_results.csv', '')
                species_part = base_name.split('4mC_')[-1]
                species = species_part.replace('_', ' ').title()
                
                df = pd.read_csv(os.path.join(project_dir, file))
                df['species'] = species
                all_dfs.append(df)
            except Exception as e:
                print(f"skip processing the file {file}: {str(e)}")
                continue
    
    if not all_dfs:
        raise ValueError("No valid data file found! Please check the file format and directory path")
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    baseline = full_df[full_df['combination'] == 'full']
    if baseline.empty:
        missing_species = full_df['species'].unique()
        raise ValueError(f"Lack of baseline data（combination='full'），missing species：{missing_species}")

    merged = full_df.merge(baseline, on='species', suffixes=('', '_baseline'))
    
    metrics = ['ACC', 'SN', 'SP', 'F1', 'MCC', 'AUC']
    for metric in metrics:
        merged[f'{metric}_change'] = (merged[metric] - merged[f'{metric}_baseline']) / merged[f'{metric}_baseline']
        merged[f'{metric}_change'] = merged[f'{metric}_change'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return merged[merged['combination'] != 'full'], metrics

def plot_ablation_analysis(df, metrics, base_models):
    """最终优化版可视化函数"""
    species_list = sorted(df['species'].unique())

    model_colors = {
        'CNN': '#2ecc71',
        'BLSTM': '#e74c3c',
        'Transformer': '#3498db'
    }

    fig_width = 24
    fig_height = 12
    bar_width = 0.25
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = fig.subplots(2, 3).flatten()
    
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        all_values = []
        for model in base_models:
            model_data = df[df['combination'] == model]
            values = []
            for species in species_list:
                subset = model_data[model_data['species'] == species]
                values.append(subset[f'{metric}_change'].values[0] if not subset.empty else 0)
            all_values.append(values)

        x_base = np.arange(len(species_list))
        for i, model in enumerate(base_models):
            x_offset = x_base + (i - 1) * bar_width
            ax.bar(x_offset, all_values[i], 
                   width=bar_width, 
                   color=model_colors[model],
                   edgecolor='white',
                   linewidth=1.2,
                   alpha=0.95,
                   label=model)

        ax.set_title(metric, pad=12)
        ax.axhline(0, color='#444444', linewidth=1.2)

        ax.tick_params(
            axis='y',
            which='both',
            length=3,
            left=True,
            right=False
        )

        ax.yaxis.grid(True, linestyle=':', alpha=0.3)

        ax.set_xticks(x_base)

        if idx >= 3:
            ax.set_xticklabels(
                species_list, 
                rotation=35,
                ha='right',
                rotation_mode='anchor'
            )
            ax.tick_params(
                axis='x',
                bottom=True,
                top=False,
                labelbottom=True
            )
        else:
            ax.set_xticklabels([])
            ax.tick_params(
                axis='x',
                bottom=True,
                top=False,
                labelbottom=False
            )

        flat_values = [v for sublist in all_values for v in sublist]
        max_val = max(flat_values, default=0)
        min_val = min(flat_values, default=0)
        margin = max(abs(max_val), abs(min_val)) * 0.35
        ax.set_ylim(min_val - margin, max_val + margin)

    legend_handles = [
        plt.Rectangle((0,0),1,1, 
                     color=model_colors[model],
                     ec='white',
                     label=model) 
        for model in base_models
    ]
    fig.legend(
        handles=legend_handles,
        loc='center right',
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        title="Models",
        frameon=True,
        framealpha=0.9,
        borderaxespad=0.5
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.93])
    
    output_path = os.path.join(output_dir, 'ablation_combined.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"successfully generates: {output_path}")

if __name__ == "__main__":
    try:
        processed_data, metric_list = load_and_process_data()
        base_models = ['CNN', 'BLSTM', 'Transformer']
        plot_ablation_analysis(processed_data, metric_list, base_models)
        print(f"All results have been saved to: {output_dir}")
    except Exception as e:
        print(f"Error: {str(e)}")