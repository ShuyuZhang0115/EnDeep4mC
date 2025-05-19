import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# 数据准备
models = ['4mcPred-SVM', '4mCCNN', 'Deep4mC', 'Hyb4mC', 'DeepSF-4mC', 'EpiTEAmDNA', 'Ours']
metrics = ['Sn', 'Sp', 'Acc', 'MCC', 'AUC', 'F1']

data = {
    '4mcPred-SVM': [0.6020, 0.6500, 0.5530, np.nan, 0.632, 0.6210],
    '4mCCNN': [0.6830, 0.7850, 0.5810, np.nan, 0.6830, 0.7120],
    'Deep4mC': [0.7750, 0.7430, 0.8080, np.nan, 0.8580, 0.7680],
    'Hyb4mC': [0.8730, 0.8950, 0.8510, np.nan, 0.9460, 0.8760],
    'DeepSF-4mC': [0.861, 0.880, 0.842, 0.723, 0.861, 0.864],
    'EpiTEAmDNA': [0.8845, 0.9013, 0.8677, 0.7695, 0.9527, 0.8864],
    'Ours': [0.913323, 0.917255, 0.909391, 0.826677, 0.969554, 0.913662]
}

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # 闭合图形

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 'red']
model_colors = dict(zip(models, colors))

for model in models:
    values = data[model]
    filled_values = [v if not np.isnan(v) else 0 for v in values]
    filled_values += [filled_values[0]]  # 闭合图形
    ax.plot(angles, filled_values, 'o-', linewidth=2, label=model, color=model_colors[model])
    ax.fill(angles, filled_values, alpha=0.1, color=model_colors[model])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_yticklabels([])
ax.set_ylim(0, 1)

plt.title('Model Performance Comparison on 4mC_A.thaliana Dataset', pad=20, fontsize=14)
legend_elements = [Patch(facecolor=model_colors[model], label=model) for model in models]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.1, 1.1), loc='upper left')

for i, metric in enumerate(metrics):
    for model in models:
        value = data[model][i]
        if not np.isnan(value):
            angle = angles[i]
            r = value + 0.03
            ax.text(angle, r, f'{value:.3f}', ha='center', va='center', 
                   fontsize=8, color=model_colors[model])

save_dir = "/your_path/EnDeep4mC/evaluations"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "radar_plot.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {save_path}")

plt.show()