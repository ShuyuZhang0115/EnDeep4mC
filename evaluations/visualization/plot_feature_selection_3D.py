import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import os

csv_path = "/your_path/EnDeep4mC/evaluations/workflow_plot/optimal_feature_subsets.csv"
output_dir = "/your_path/EnDeep4mC/evaluations/workflow_plot"

os.makedirs(output_dir, exist_ok=True)

results = pd.read_csv(csv_path)

all_features = [
    "ENAC", "binary", "NCP", "EIIP", "Kmer4", "CKSNAP", "PseEIIP",
    "TNC", "RCKmer5", "SCPseTNC", "PCPseTNC", "ANF", "NAC", "TAC"
]
SPECIES = [
    '4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster',
    '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii'
]
MODELS = ['CNN', 'BLSTM', 'Transformer']

colors = ['#3498db', '#e74c3c']
cmap = ListedColormap(colors)

plt.figure(figsize=(20, 15))
ax = plt.subplot(111, projection='3d')

# Plane tilt parameters
tilt_angle = 45
offset = 1.0

X, Y = np.meshgrid(np.arange(len(all_features)), np.arange(len(SPECIES)))

for model_idx, model_name in enumerate(MODELS):
    feature_matrix = np.zeros((len(SPECIES), len(all_features)))
    model_data = results[results['Model'] == model_name]
    
    for i, species in enumerate(SPECIES):
        species_data = model_data[model_data['Species'] == species]
        if not species_data.empty:
            features = [f.strip() for f in species_data['Top_Features'].iloc[0].split(",")]
            feature_matrix[i] = [1 if feat in features else 0 for feat in all_features]

    Z = np.full_like(X, model_idx * offset)

    x_coords = X + (model_idx * offset) * np.cos(np.radians(tilt_angle))
    y_coords = Y + (model_idx * offset) * np.sin(np.radians(tilt_angle))
    z_coords = Z

    surf = ax.plot_surface(x_coords, y_coords, z_coords, 
                          facecolors=cmap(feature_matrix),
                          rstride=1, cstride=1,
                          shade=False,
                          alpha=0.9,
                          edgecolor='white', 
                          linewidth=0.3)

    ax.text(x=len(all_features)+1, 
           y=len(SPECIES)+1,
           z=model_idx*offset,
           s=model_name,
           fontsize=14,
           ha='center')

ax.set_xticks(np.arange(len(all_features)))
ax.set_xticklabels(all_features, rotation=45, ha='right', fontsize=12)
ax.set_yticks(np.arange(len(SPECIES)))
ax.set_yticklabels([s.split('_')[1] for s in SPECIES], fontsize=12)
ax.set_zticks([0, offset, 2*offset])
ax.set_zticklabels([])

ax.view_init(elev=25, azim=-45)
ax.set_box_aspect([3, 2, 1])

ax.set_xlabel('Feature Encodings', fontsize=20, labelpad=60)
ax.set_ylabel('Species', fontsize=20, labelpad=20)
ax.set_zlabel('Models', fontsize=20, labelpad=0)

ax.grid(True)
ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = True

plt.tight_layout()
plot_path = os.path.join(output_dir, "3D_Feature_Usage.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"The 3D feature selection space map has been saved to: {plot_path}")