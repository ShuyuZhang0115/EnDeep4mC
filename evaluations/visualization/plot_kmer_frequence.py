import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_combined_comparison(csv_path, output_path, top_n=10, figsize=(24, 8)):
    """
    Generate merged comparison chart (1x3 layout)
    :param csv_path: path to significant_kmers.csv
    :param output_path: Output image path
    :param top_n: Each k value displays the first n significant kmers
    :param figsize: Canvas size (width, height)
    """
    # Read data
    df = pd.read_csv(csv_path)
    
    # Create canvas and style settings
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    palette = {'prok': '#e41a1c', 'euk': '#377eb8'}  # Maintain color consistency
    plt.rc('font', size=12)  # Unified font size
    
    # Create subgraph layout
    k_values = sorted(df['k'].unique())
    fig, axes = plt.subplots(1, len(k_values), figsize=figsize)
    if len(k_values) == 1:  # Dealing with a single k value
        axes = [axes]
    
    # Traverse each k value and draw a subgraph
    for idx, (k, ax) in enumerate(zip(k_values, axes)):
        # Filter and sort data
        k_data = df[df['k'] == k].copy()
        k_data = k_data[k_data['p_value'] < 0.05].sort_values('p_value').head(top_n)
        if k_data.empty:
            ax.set_visible(False)
            continue
        
        # Prepare drawing data
        plot_data = pd.melt(
            k_data,
            id_vars=['kmer'],
            value_vars=['prok_freq', 'euk_freq'],
            var_name='group',
            value_name='frequency'
        )
        plot_data['group'] = plot_data['group'].str.replace('_freq', '')
        
        # Draw the bar chart
        sns.barplot(
            x='kmer',
            y='frequency',
            hue='group',
            data=plot_data,
            ax=ax,
            palette=palette,
            saturation=0.8
        )
        
        # decoration for subfigure
        ax.set_title(f'k = {k}', fontsize=14, pad=12)
        ax.set_xlabel('')
        ax.set_ylabel('' if idx > 0 else 'Frequency Difference (pos-neg)')  # Only display the y-axis label for the first one
        
        # Automatically calculate the display ratio for different k value ranges
        freq_values = plot_data['frequency']
        abs_max = freq_values.abs().max()
        
        # Dynamically set y-axis range
        if abs_max == 0:
            abs_max = 0.001  # Prevent zeroing
        buffer = abs_max * 0.2  # Leave 20% blank space
        
        # Automatically select proportion range
        upper = max(freq_values.max() + buffer, abs_max*0.3)  # Ensure small changes are visible
        lower = min(freq_values.min() - buffer, -abs_max*0.3)
        
        ax.set_ylim(lower, upper)
        
        # Adjust the scale label
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add intelligent numerical annotation
        for p in ax.patches:
            height = p.get_height()
            if abs(height) < 1e-5:  # Ignore minor changes
                continue
            
            # Dynamically select annotation format
            current_range = upper - lower  # The actual display range of the current subgraph
            if current_range > 0.1:
                # Large span using conventional decimal format
                text = f"{height:.3f}"
                fs = 9
            elif current_range > 0.01:
                # Medium span using mixed format
                text = f"{height:.2e}".replace('e-0', 'e-').replace('e+0', 'e+') 
                fs = 8
            else:
                # Small span forced Scientific notation
                text = f"{height:.1e}".replace('e-0', 'e-').replace('e+0', 'e+')
                fs = 8
            
            # Positive and negative sign processing
            if height > 0:
                text = '+' + text.lstrip('+')
                va_pos = 'bottom'
                y_offset = current_range * 0.03  # dynamic offset
            else:
                va_pos = 'top'
                y_offset = -current_range * 0.03
            
            # Adaptive color contrast
            text_color = 'black' if current_range < 0.05 else 'white'
            
            ax.annotate(
                text,
                (p.get_x() + p.get_width()/2, height + y_offset),
                ha='center',
                va=va_pos,
                fontsize=fs,
                color=text_color,
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='#404040' if text_color=='white' else 'white',
                    alpha=0.8,
                    edgecolor='none'
                )
            )
        
        # Remove duplicate legends
        if idx > 0:
            ax.get_legend().remove()
        else:
            ax.legend_.set_title('')

    # Add global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        ['Prokaryote', 'Eukaryote'],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        fontsize=12,
        frameon=False
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space at the bottom for the legend
    
    # Save charts
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The merged chart has been saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate merged k-mer comparison chart')
    parser.add_argument('-i', '--input', required=True, help='Path to significant_kmers.csv')
    parser.add_argument('-o', '--output', default='merged_comparison.png', help='Output image path')
    parser.add_argument('-n', '--top', type=int, default=10, help='Display the first N kmers for each k value')
    
    args = parser.parse_args()
    
    visualize_combined_comparison(
        csv_path=args.input,
        output_path=args.output,
        top_n=args.top
    )