import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# configs
BASE_MODELS = ['CNN', 'BLSTM', 'Transformer']
DATASETS = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster',
           '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']
CLEAN_DATASETS = [ds.replace("4mC_", "") for ds in DATASETS]  # 清洗后的数据集名称
PROJECT_DIR = Path("/your_path/EnDeep4mC")
SAVE_DIR = PROJECT_DIR / "evaluations/ablation_feature"
OUTPUT_DIR = SAVE_DIR / "processed_visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess(model_name):
    """Load and preprocess the results of a single model"""
    df = pd.read_csv(SAVE_DIR / f"ablation_{model_name}.csv", index_col=0)
    
    # Fill in missing values as 0 and ensure column consistency
    all_features = set()
    for ds in DATASETS:
        all_features.update(df.columns.tolist())
    
    # Rebuilding a complete DataFrame
    full_df = pd.DataFrame(index=DATASETS, columns=list(all_features))
    for ds in DATASETS:
        if ds in df.index:
            ds_values = df.loc[ds].to_dict()
            full_df.loc[ds] = [ds_values.get(f, 0) for f in full_df.columns]
        else:
            full_df.loc[ds] = 0
    
    return full_df.fillna(0)

def calculate_model_averages():
    """Calculate species level averages across models"""
    model_dfs = {}
    for model in BASE_MODELS:
        model_dfs[model] = load_and_preprocess(model)
    
    all_features = set()
    for df in model_dfs.values():
        all_features.update(df.columns)
    all_features = sorted(all_features)
    
    avg_df = pd.DataFrame(0, index=DATASETS, columns=all_features)
    
    for ds in DATASETS:
        for feat in all_features:
            values = []
            for model in BASE_MODELS:
                df = model_dfs[model]
                values.append(df.loc[ds, feat] if feat in df.columns else 0)
            avg_df.loc[ds, feat] = np.mean(values)
    
    return avg_df

def plot_improved_heatmap(avg_df):
    """Symmetric color gradient heatmap and save matrix"""
    plt.figure(figsize=(16, 10))

    abs_max = max(abs(avg_df.min().min()), abs(avg_df.max().max()))
    vmin = -abs_max
    vmax = abs_max

    colors = [
        (0.0, "#8B0000"),
        (0.5, "#FFFFFF"),
        (1.0, "#0066CC")
    ]
    cmap = LinearSegmentedColormap.from_list("custom_div", colors)

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    sorted_features = avg_df.mean(axis=0).sort_values(ascending=False).index.tolist()
    plot_df = avg_df[sorted_features]

    save_matrix = plot_df.T
    save_matrix.to_csv(OUTPUT_DIR / "feature_impact_matrix.csv", 
                      float_format="%.6f")

    ax = sns.heatmap(
        plot_df.T,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt=".4f",
        linewidths=0.5,
        annot_kws={
            "size": 9,
            "weight": "bold"  # 加粗注释文字
        },
        cbar_kws={
            'label': 'Average Accuracy Change',
            'ticks': [vmin, 0, vmax]
        }
    )

    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels([f"{vmin:.4f}", "0", f"{vmax:.4f}"])
    cbar.ax.tick_params(labelsize=9)

    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_weight("bold")
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.yaxis.label.set_weight('bold')  # 加粗颜色条标题
    
    plt.title(
        "Cross-Model Feature Ablation Impact (Averaged)",
        pad=20,
        fontsize=14,
        weight='bold'
    )
    plt.xlabel(
        "Dataset",
        fontsize=12,
        weight='bold'
    )
    plt.ylabel(
        "Features",
        fontsize=12,
        weight='bold'
    )

    ax.set_xticklabels(
        CLEAN_DATASETS,
        rotation=45,
        ha="right",
        fontsize=10,
        weight='bold'
    )

    for label in ax.get_yticklabels():
        label.set_weight("bold")
        label.set_size(10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_impact_heatmap.png", dpi=300)
    plt.close()

def plot_faceted_barchart(avg_df):
    """Maintain the two-color scheme for the facet map"""
    plt.figure(figsize=(18, 8))
    
    plot_df = avg_df.reset_index().melt(id_vars='index', var_name='Feature', value_name='Impact')
    plot_df.columns = ['Dataset', 'Feature', 'Impact']
    plot_df['Dataset'] = plot_df['Dataset'].str.replace('4mC_', '')
    
    feature_order = avg_df.mean().sort_values(ascending=False).index.tolist()

    g = sns.FacetGrid(
        plot_df, 
        col='Dataset',
        col_wrap=3, 
        height=3.8, 
        sharey=False, 
        despine=False,
        col_order=CLEAN_DATASETS
    )
    
    def draw_subplot(data, **kwargs):
        current_data = data.set_index('Feature').reindex(feature_order).reset_index()
        current_data = current_data.dropna(subset=['Impact'])
        colors = ['#d7191c' if x < 0 else '#2c7bb6' for x in current_data['Impact']]
        ax = sns.barplot(
            data=current_data,
            x='Impact',
            y='Feature',
            order=feature_order,
            palette=colors,
            dodge=False
        )
        ax.axvline(0, color='black', linewidth=0.5, linestyle='-', zorder=3)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.set_xlabel('', weight='bold')
        ax.set_ylabel('', weight='bold')

        ax.tick_params(
            axis='x',
            labelsize=15,
            width=2,
            which='major',
            labelrotation=0
        )
        for label in ax.get_xticklabels():
            label.set_weight("bold")

        ax.tick_params(
            axis='y',
            labelsize=15,
            width=2,
            which='major'
        )
        for label in ax.get_yticklabels():
            label.set_weight("bold")
    
    g.map_dataframe(draw_subplot)

    g.set_titles("{col_name}", size=16, weight='bold')  # 加粗子图标题

    for i, ax in enumerate(g.axes.flat):
        if i % 3 != 0:
            ax.set_yticklabels([])
    
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "faceted_barchart.png", dpi=300)
    plt.close()

def main():
    avg_df = calculate_model_averages()
    avg_df.to_csv(OUTPUT_DIR / "cross_model_average.csv")
    
    plot_improved_heatmap(avg_df)
    plot_faceted_barchart(avg_df)
    
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()