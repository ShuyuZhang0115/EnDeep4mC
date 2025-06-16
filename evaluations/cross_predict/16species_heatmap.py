import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# ================== 配置参数 ==================
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

# ================== 数据预处理 ==================
def clean_species_names(df):
    """统一处理行列名称"""
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

# ================== 可视化参数 ==================
colors = ["#FFDEDE", "#FF6B6B", "#C70000"]
cmap_reds = LinearSegmentedColormap.from_list("custom_reds", colors, N=256)

palette = {
    "Plant": "#2ecc71",
    "Animal": "#3498db",
    "Microorganism": "#e74c3c"
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,  # 基础字体大小（影响图例）
    'axes.titlesize': 24,  # 标题字体（已删除）
    'axes.labelsize': 18,  # 坐标轴标签
    'xtick.labelsize': 18,  # X轴刻度
    'ytick.labelsize': 18,  # Y轴刻度
    'pdf.fonttype': 42
})

# ================== 单个热图生成函数 ==================
def generate_single_heatmap(model_name, data, output_path):
    """生成并保存单个模型的热图"""
    fig = plt.figure(figsize=(36, 24))  # 宽度增加60%，高度增加70%
    ax = fig.add_subplot(111)
    
    plot_data = data.drop(columns=['Category']).astype(float)
    
    heatmap = sns.heatmap(
        plot_data,
        cmap=cmap_reds,
        annot=True,
        fmt=".3f",
        annot_kws={
            "size": 28,  # 热图数值字体大小
            "weight": "bold",
            "color": "black"  # 修改1：数值颜色改为黑色
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
    
    # 设置坐标轴标签（修改2：删除标题）
    ax.set_xlabel(
        "Feature Encoding Methods",
        labelpad=15,
        fontsize=50,  # X轴标签字体大小
        fontweight='bold'
    )
    ax.set_ylabel(
        "Species",
        labelpad=20,
        fontsize=50,  # Y轴标签字体大小
        fontweight='bold'
    )
    
    # Y轴标签样式
    for label in ax.get_yticklabels():
        species = label.get_text()
        label.set_color(palette[species_classification[species]])
        label.set_fontsize(40)  # Y轴刻度字体大小
        label.set_weight('bold')
    
    # X轴标签样式
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        fontsize=40,  # X轴刻度字体大小
        weight='bold'
    )
    
    # 颜色条设置
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)  # 颜色条刻度字体
    cbar.set_label(
        label='Accuracy',
        rotation=90,
        size=40,  # 颜色条标签字体
        weight='bold',
        labelpad=25
    )
    
    # 调整布局参数
    plt.subplots_adjust(
        top=0.88,
        bottom=0.18,
        left=0.12,
        right=0.85
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"生成文件: {output_path}")

# ================== 图例生成函数 ==================
def generate_legend(output_path):
    """单独生成分类图例"""
    fig = plt.figure(figsize=(8, 1))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # 创建图例元素
    legend_elements = [Patch(facecolor=v, edgecolor=v, label=k) for k,v in palette.items()]
    
    # 绘制图例（保持原始设置）
    leg = fig.legend(
        handles=legend_elements,
        loc='center',
        ncol=3,
        frameon=False,
        fontsize=30,  # 图例文本大小
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.5
    )
    
    # 调整布局
    plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"生成图例文件: {output_path}")

# ================== 主执行逻辑 ==================
if __name__ == "__main__":
    output_dir = "/home/zqzhangshuyu/Projs/EnDeep4mC-V2/evaluations/acc_heatmap"
    
    # 生成所有热图
    for model_name, data in processed_data.items():
        output_path = f"{output_dir}/{model_name}_Final_Heatmap.pdf"
        generate_single_heatmap(model_name, data, output_path)
    
    # 单独生成图例（修改3）
    generate_legend(f"{output_dir}/Species_Category_Legend.png")
    
    print("\n所有热图生成完成，保存路径:", output_dir)