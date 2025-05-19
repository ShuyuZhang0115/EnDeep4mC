import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import ticker

# ======================
# Configs
# ======================
BASE_DIR = os.path.expanduser("~/Projs/EnDeep4mC/")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluations", "feature_compare")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMN_MAPPING = {
    'CNN': {'CNN_Test_ACC':'ACC', 'CNN_SN':'SN', 'CNN_SP':'SP', 'CNN_F1':'F1', 'CNN_MCC':'MCC', 'CNN_AUC':'AUC'},
    'BLSTM': {'BLSTM_Test_ACC':'ACC', 'BLSTM_SN':'SN', 'BLSTM_SP':'SP', 'BLSTM_F1':'F1', 'BLSTM_MCC':'MCC', 'BLSTM_AUC':'AUC'},
    'Transformer': {'Transformer_Test_ACC':'ACC', 'Transformer_SN':'SN', 'Transformer_SP':'SP', 'Transformer_F1':'F1', 'Transformer_MCC':'MCC', 'Transformer_AUC':'AUC'},
    'Ensemble': {'accuracy':'ACC', 'sn':'SN', 'sp':'SP', 'f1':'F1', 'mcc':'MCC', 'auc':'AUC'}
}

DATASET_ORDER = [
    'A.thaliana',
    'C.elegans',
    'D.melanogaster',
    'E.coli',
    'G.subterraneus',
    'G.pickeringii'
]

# ======================
# Data processing function
# ======================
def load_and_process(model):
    try:
        file_name = "ensemble_indiv_results.csv" if model == "Ensemble" else f"{model.lower()}_auto_summary.csv"
        dfs_path = os.path.join(BASE_DIR, "pretrained_models/indiv", file_name)
        full_path = os.path.join(BASE_DIR, "pretrained_models/indiv_14_feature", file_name)
        
        dfs_df = pd.read_csv(dfs_path).rename(columns=COLUMN_MAPPING[model]).assign(Method='DFS')
        full_df = pd.read_csv(full_path).rename(columns=COLUMN_MAPPING[model]).assign(Method='Full')
        
        combined = pd.concat([dfs_df, full_df], ignore_index=True)
        combined['Dataset'] = combined['Dataset'].str.replace('4mC_', '')
        
        combined['Dataset'] = pd.Categorical(
            combined['Dataset'], 
            categories=DATASET_ORDER,
            ordered=True
        )
        
        metrics_order = ['ACC', 'SN', 'SP', 'F1', 'MCC', 'AUC']
        melted = combined.melt(
            id_vars=['Dataset', 'Method'],
            value_vars=metrics_order,
            var_name='Metric',
            value_name='Value'
        )
        melted['Metric'] = pd.Categorical(melted['Metric'], categories=metrics_order, ordered=True)
        
        return melted.dropna().sort_values(['Dataset', 'Metric'])

    except Exception as e:
        print(f"处理错误: {str(e)}")
        return None

# ======================
# Viz function
# ======================
def plot_comparison(data, model):
    plt.figure(figsize=(18, 18))
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'DejaVu Sans',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
    })
    
    g = sns.FacetGrid(
        data,
        col="Dataset",
        col_wrap=2,
        col_order=DATASET_ORDER,
        height=8,
        aspect=1.2,
        sharey=False,
        sharex=False,
        despine=False,
        gridspec_kws={'wspace':0.25, 'hspace':0.5}
    )
    
    plot_params = {
        'linewidth': 4,
        'markersize': 10,
        'markeredgewidth': 2
    }
    
    def plot_lines(**kwargs):
        data = kwargs.pop("data")
        ax = plt.gca()
        
        sns.lineplot(
            data=data,
            x="Metric",
            y="Value",
            hue="Method",
            style="Method",
            markers={'DFS': 'X', 'Full': 'o'},
            dashes=False,
            palette={'DFS': '#E74C3C', 'Full': '#2E86C1'},
            **plot_params,
            ax=ax,
            sort=False
        )
        
        pivot = data.pivot_table(index='Metric', columns='Method', values='Value')
        pivot['delta'] = pivot['DFS'] - pivot['Full']
        
        for idx, metric in enumerate(pivot.index):
            delta = pivot.loc[metric, 'delta']
            ax.text(
                idx, 
                max(pivot.loc[metric, 'DFS'], pivot.loc[metric, 'Full']) + 0.02,
                f"{delta:+.3f}",
                ha='center',
                va='bottom',
                fontsize=20,
                fontweight='bold',
                color='#2C3E50',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
            )
        
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(30)
        
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(
            pivot.index.tolist(),
            rotation=35,
            ha='right',
            fontsize=30,
            fontweight='bold'
        )
        
        ax.set_xlabel("", fontsize=12, fontweight='bold')
        ax.set_ylabel("", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        y_padding = data['Value'].max() - data['Value'].min()
        ax.set_ylim(
            data['Value'].min() - y_padding*0.1,
            data['Value'].max() + y_padding*0.2
        )
        
        if ax.get_legend():
            ax.get_legend().remove()

    g.map_dataframe(plot_lines)
    
    for idx, ax in enumerate(g.axes.flat):
        if idx < 4:
            ax.set_xticklabels([])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontweight('bold')

    handles = [
        plt.Line2D([], [], color='#E74C3C', marker='X', linestyle='-', 
                  markersize=10, markeredgewidth=2, label='DFS'),
        plt.Line2D([], [], color='#2E86C1', marker='o', linestyle='-', 
                  markersize=10, markeredgewidth=2, label='Full')
    ]
    if model != 'Ensemble':
        g.fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            fontsize=30,
            prop={'weight': 'bold'},
            title_fontsize=12
        )
    
    for ax, title in zip(g.axes.flat, DATASET_ORDER):
        ax.set_title(
            title, 
            fontsize=30,
            pad=10,
            y=1,
            color='#34495E',
            fontweight='bold'
        )
    
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{model}_line_comparison.png"),
        bbox_inches='tight',
        dpi=600,
        facecolor='white'
    )
    plt.close()

    if model == 'Ensemble':
        plt.figure(figsize=(12, 1))
        plt.axis('off')
        legend = plt.legend(
            handles=handles,
            loc='center',
            ncol=2,
            frameon=False,
            fontsize=40,
            prop={'weight':'bold'}
        )
        plt.savefig(
            os.path.join(OUTPUT_DIR, "feature_legend.png"),
            bbox_inches='tight',
            dpi=600,
            facecolor='white'
        )
        plt.close()

# ======================
# Main execution logic
# ======================
if __name__ == "__main__":
    models = ['CNN', 'BLSTM', 'Transformer', 'Ensemble']
    for model in models:
        data = load_and_process(model)
        if data is not None:
            plot_comparison(data, model)
    print(f"\n处理完成，结果保存在: {OUTPUT_DIR}")