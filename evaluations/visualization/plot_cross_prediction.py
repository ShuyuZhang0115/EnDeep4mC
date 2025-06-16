import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ================== Configs ==================
PROJECT_DIR = "/your_path/EnDeep4mC/evaluations/cross_predict"
ACC_CSV_PATH = os.path.join(PROJECT_DIR, "accuracy_matrix.csv")
AUC_CSV_PATH = os.path.join(PROJECT_DIR, "auc_matrix.csv")
OUTPUT_DIR = "/your_path/EnDeep4mC/evaluations/cross_predict/visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== Dataset name formatting ==================
def clean_dataset_names(df):
    df.columns = df.columns.str.replace('4mC_', '')
    df.index = df.index.str.replace('4mC_', '')
    return df

# ================== Classification Configuration ==================
SPECIES_CATEGORIES = {
    'Plant': {
        'color': '#2ecc71',
        'datasets': ['A.thaliana2', 'C.equisetifolia', 'F.vesca']
    },
    'Animal': {
        'color': '#3498db',
        'datasets': ['C.elegans2', 'D.melanogaster2']
    },
    'Microbe': {
        'color': '#e74c3c',
        'datasets': ['E.coli2', 'G.subterraneus2', 
                    'G.pickeringii2', 'S.cerevisiae', 'Tolypocladium']
    }
}

CATEGORY_ORDER = ['Plant', 'Animal', 'Microbe']
ORDERED_DATASETS = []
for category in CATEGORY_ORDER:
    ORDERED_DATASETS.extend(SPECIES_CATEGORIES[category]['datasets'])

# ================== Customize color mapping ==================
colors = ["#FFDEDE", "#FF6B6B", "#C70000"]
cmap_reds = LinearSegmentedColormap.from_list("custom_reds", colors, N=256)

# ================== Heatmap generation function ==================
def generate_enhanced_heatmap(matrix_path, metric_name):

    matrix = pd.read_csv(matrix_path, index_col=0)
    matrix = clean_dataset_names(matrix)
    matrix = matrix.reindex(index=ORDERED_DATASETS, columns=ORDERED_DATASETS)

    plt.figure(figsize=(28, 24))
    ax = sns.heatmap(
        matrix.astype(float), 
        annot=True, 
        fmt=".3f",
        cmap=cmap_reds,
        cbar_kws={'label': metric_name, 'shrink': 0.6, 'aspect': 20},
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        annot_kws={
            "size": 35,
            "weight": "bold",
            "color": "black"
        },
        square=True
    )

    ax.invert_yaxis()
    ax.xaxis.tick_bottom()

    plt.xlabel('Test Datasets', fontsize=60, labelpad=30, weight='bold', family='sans-serif')
    plt.ylabel('Train Datasets', fontsize=60, labelpad=40, weight='bold', family='sans-serif')

    ax.set_xticks(np.arange(len(ORDERED_DATASETS)) + 0.5)
    ax.set_xticklabels(
        ORDERED_DATASETS, 
        rotation=90,
        ha='center',
        va='top',
        fontsize=50,
        fontweight='bold',
        family='sans-serif',
        color='black'
    )

    ax.set_yticks(np.arange(len(ORDERED_DATASETS)) + 0.5)
    ax.set_yticklabels(
        ORDERED_DATASETS,
        rotation=0,
        fontsize=50,
        fontweight='bold',
        family='sans-serif',
        color='black'
    )

    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)

    def set_label_colors(labels):
        for label in labels:
            text = label.get_text()
            for category, info in SPECIES_CATEGORIES.items():
                if text in info['datasets']:
                    label.set_color(info['color'])
                    label.set_fontsize(50)
                    break
    
    set_label_colors(ax.get_xticklabels())
    set_label_colors(ax.get_yticklabels())

    def draw_category_lines():
        accum_idx = 0
        for category in CATEGORY_ORDER:
            n = len(SPECIES_CATEGORIES[category]['datasets'])
            accum_idx += n
            ax.axhline(y=accum_idx, color='black', linewidth=4, linestyle='--')
            ax.axvline(x=accum_idx, color='black', linewidth=4, linestyle='--')
    
    draw_category_lines()

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=40)  
    cbar.set_label(label=metric_name, size=60, weight='bold', family='sans-serif')  

    png_path = os.path.join(OUTPUT_DIR, f'{metric_name.lower()}_heatmap.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generate file: {png_path}")

# ================== Main execution logic ==================
if __name__ == "__main__":
    generate_enhanced_heatmap(ACC_CSV_PATH, "ACC")
    generate_enhanced_heatmap(AUC_CSV_PATH, "AUC")
    print("\nAll heat maps have been generated, save path:", os.path.abspath(OUTPUT_DIR))