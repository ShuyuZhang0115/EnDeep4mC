import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

ABLATION_PALETTE = [
    '#2E5C87', '#8A4F7D', '#C05546', '#4B8B3B',
    '#D4A53F', '#6F7C85', '#7F3C8D', '#11A579',
    '#3969AC', '#F2B701', '#E73F74', '#80BA5A',
    '#E68310', '#008695'
]

FEATURES = ['ENAC', 'binary', 'NCP', 'EIIP', 'Kmer4', 'CKSNAP', 
           'PseEIIP', 'TNC', 'RCKmer5', 'SCPseTNC', 'PCPseTNC', 
           'ANF', 'NAC', 'TAC']
FEATURE_COLORS = {feat: ABLATION_PALETTE[i] for i, feat in enumerate(FEATURES)}

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'DejaVu Sans',
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.bbox': 'tight'
})

project_dir = '/your_path/EnDeep4mC'
SAVE_DIR = os.path.join(project_dir, 'evaluations', 'ablation_feature', 'feature_percentage')
os.makedirs(SAVE_DIR, exist_ok=True)

DATASETS = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster',
           '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']
MODELS = ['CNN', 'BLSTM', 'Transformer', 'Ensemble']

def load_usage_data():
    usage_data = {}
    
    for model in MODELS[:-1]:
        usage_data[model] = {}
        rank_file = os.path.join(project_dir, 'feature_engineering', 'fea_index', f'{model}_Ranking.csv')
        acc_file = os.path.join(project_dir, 'feature_engineering', 'ifs_result', f'{model}_Feature_Acc_Table.csv')
        
        rankings = pd.read_csv(rank_file, index_col=0)
        best_n = pd.read_csv(acc_file, index_col=0).loc['best_n']
        
        for dataset in DATASETS:
            n = int(best_n[dataset])
            used_features = rankings.loc[dataset].values[:n]
            for f in FEATURES:
                usage_data[model][f] = usage_data[model].get(f, 0) + (1 if f in used_features else 0)
    
    usage_data['Ensemble'] = {}
    for f in FEATURES:
        count = 0
        for model in MODELS[:-1]:
            count += usage_data[model][f]
        usage_data['Ensemble'][f] = count
    
    for model in MODELS[:-1]:
        total = len(DATASETS)
        for f in FEATURES:
            usage_data[model][f] = (usage_data[model][f] / total) * 100
            
    return usage_data

def plot_usage_percentage(usage_data):
    fig = plt.figure(figsize=(18, 14))
    axes = [plt.subplot(2, 2, i+1) for i in range(4)]

    for ax, model in zip(axes[:3], MODELS[:-1]):
        df = pd.DataFrame.from_dict(usage_data[model], orient='index', columns=['Percentage'])
        df = df.sort_values(by='Percentage', ascending=False)

        color_list = [FEATURE_COLORS[feat] for feat in df.index]
        
        bars = sns.barplot(
            x=df.index,
            y=df['Percentage'],
            ax=ax,
            palette=color_list,
            saturation=0.85,
            edgecolor='white',
            linewidth=1
        )
        ax.set_title(f'{model} Feature Usage Percentage', pad=12, fontweight='semibold')
        ax.set_ylabel('Usage Percentage (%)', fontweight='semibold')
        ax.set_ylim(0, 110)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

        for p in bars.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width()/2.,
                height + 2,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9,
                color='#444444'
            )

    df_ensemble = pd.DataFrame.from_dict(usage_data['Ensemble'], orient='index', columns=['Count'])
    df_ensemble = df_ensemble.sort_values(by='Count', ascending=False)

    ensemble_colors = [FEATURE_COLORS[feat] for feat in df_ensemble.index]
    
    bars = sns.barplot(
        x=df_ensemble.index,
        y=df_ensemble['Count'],
        ax=axes[3],
        palette=ensemble_colors,
        saturation=0.85,
        edgecolor='white',
        linewidth=1
    )
    axes[3].set_title('Ensemble Feature Usage Count', pad=12, fontweight='semibold')
    axes[3].set_ylabel('Times Used in Base Models', fontweight='semibold')

    for p in bars.patches:
        height = p.get_height()
        axes[3].text(
            p.get_x() + p.get_width()/2.,
            height + 0.1,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=9,
            color='#444444'
        )

    for ax in axes:
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        for spine in ['right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#999999')
        ax.spines['bottom'].set_color('#999999')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(SAVE_DIR, 'feature_usage_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    usage_data = load_usage_data()
    plot_usage_percentage(usage_data)