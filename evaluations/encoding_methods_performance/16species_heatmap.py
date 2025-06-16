import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# ================== configuration parameter ==================
species_classification = {
    "A.thaliana": "Plant",
    "A.thaliana2": "Plant",
    "C.equisetifolia": "Plant",
    "F.vesca": "Plant",
    "C.elegans": "Animal",
    "C.elegans2": "Animal",
    "D.melanogaster": "Animal",
    "D.melanogaster2": "Animal",
    "E.coli": "Microorganism",
    "E.coli2": "Microorganism",
    "G.subterraneus": "Microorganism",
    "G.subterraneus2": "Microorganism",
    "G.pickeringii": "Microorganism",
    "G.pickeringii2": "Microorganism",
    "S.cerevisiae": "Microorganism",
    "Tolypocladium": "Microorganism"
}

model_config = [
    ("CNN", [
        "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/evaluations/acc_matrix/CNN_ACC_Matrix.csv",
        "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/evaluations/acc_matrix_extra/CNN_ACC_Matrix.csv"
    ]),
    ("BLSTM", [
        "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/evaluations/acc_matrix/BLSTM_ACC_Matrix.csv",
        "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/evaluations/acc_matrix_extra/BLSTM_ACC_Matrix.csv"
    ]),
    ("Transformer", [
        "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/evaluations/acc_matrix/Transformer_ACC_Matrix.csv",
        "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/evaluations/acc_matrix_extra/Transformer_ACC_Matrix.csv"
    ])
]

# ================== Data preprocessing ==================
def clean_species_names(df):
    """Unified processing of row and column names"""
    df.columns = [col.replace('4mC_', '') for col in df.columns.astype(str)]
    df.index = df.index.astype(str).str.replace('4mC_', '')
    return df

def load_combined_data(paths):
    combined = pd.DataFrame()
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        df = clean_species_names(df)
        combined = pd.concat([combined, df])
    
    combined['Category'] = combined.index.map(species_classification)
    combined['SortKey'] = combined['Category'].map({'Plant':0, 'Animal':1, 'Microorganism':2})
    
    return (
        combined.reset_index()
        .rename(columns={'index': 'Species'})
        .sort_values(by=['SortKey', 'Species'])
        .set_index('Species')
        .drop(columns=['SortKey'])
    )

processed_data = {model: load_combined_data(paths) for model, paths in model_config}

# ================== Visualization parameters ==================
colors = ["#FFDEDE", "#FF6B6B", "#C70000"]
cmap_reds = LinearSegmentedColormap.from_list("custom_reds", colors, N=256)

palette = {
    "Plant": "#2ecc71",
    "Animal": "#3498db",
    "Microorganism": "#e74c3c"
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,  
    'axes.titlesize': 24,  
    'axes.labelsize': 18,  
    'xtick.labelsize': 18,  
    'ytick.labelsize': 18,  
    'pdf.fonttype': 42
})

# ================== Single heatmap generation function ==================
def generate_single_heatmap(model_name, data, output_path):
    """Generate and save heatmaps of individual models"""
    fig = plt.figure(figsize=(36, 24)) 
    ax = fig.add_subplot(111)
    
    plot_data = data.drop(columns=['Category']).astype(float)
    
    heatmap = sns.heatmap(
        plot_data,
        cmap=cmap_reds,
        annot=True,
        fmt=".3f",
        annot_kws={
            "size": 28,  
            "weight": "bold",
            "color": "black"  
        },
        cbar_kws={
            'label': 'Accuracy',
            'orientation': 'vertical',
            'shrink': 0.6,
            'aspect': 20,
            'pad': 0.05
        },
        linewidths=0.8,
        ax=ax,
        vmin=0.5,
        vmax=1.0
    )
    
    # Set axis labels
    ax.set_xlabel(
        "Feature Encoding Methods",
        labelpad=15,
        fontsize=50,
        fontweight='bold'
    )
    ax.set_ylabel(
        "Species",
        labelpad=20,
        fontsize=50,
        fontweight='bold'
    )
    
    # Y-axis label style
    for label in ax.get_yticklabels():
        species = label.get_text()
        label.set_color(palette[species_classification[species]])
        label.set_fontsize(40)  
        label.set_weight('bold')
    
    # X-axis label style
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        fontsize=40,  
        weight='bold'
    )
    
    # Colorbar settings
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30) 
    cbar.set_label(
        label='Accuracy',
        rotation=90,
        size=40, 
        weight='bold',
        labelpad=25
    )
    
    # Adjust layout parameters
    plt.subplots_adjust(
        top=0.88,
        bottom=0.18,
        left=0.12,
        right=0.85
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Generate files: {output_path}")

# ================== Legend generation function ==================
def generate_legend(output_path):

    fig = plt.figure(figsize=(8, 1))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    legend_elements = [Patch(facecolor=v, edgecolor=v, label=k) for k,v in palette.items()]
    
    leg = fig.legend(
        handles=legend_elements,
        loc='center',
        ncol=3,
        frameon=False,
        fontsize=30,  
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.5
    )
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generate legend file: {output_path}")

# ================== Main execution logic ==================
if __name__ == "__main__":
    output_dir = "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/evaluations/acc_heatmap"
    
    # Generate all heatmaps
    for model_name, data in processed_data.items():
        output_path = f"{output_dir}/{model_name}_Final_Heatmap.pdf"
        generate_single_heatmap(model_name, data, output_path)
    
    # Generate legends separately
    generate_legend(f"{output_dir}/Species_Category_Legend.png")
    
    print("\nAll heat maps have been generated, save path:", output_dir)