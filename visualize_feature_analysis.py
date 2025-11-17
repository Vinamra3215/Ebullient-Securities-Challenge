import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'VB_Feature_Analysis/EBX_combined.txt'  # Path to your TXT report
OUTPUT_DIR = 'VB_Feature_Analysis/plots/'
FIGURE_DPI = 300

# ============================================================================
# SETUP
# ============================================================================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

print("="*100)
print("VB FEATURE CORRELATION VISUALIZATION")
print("="*100)
print(f"Input file: {INPUT_FILE}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# LOAD AND PARSE DATA FROM TXT FILE
# ============================================================================
print("\nLoading data from TXT report...")

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as file:
        content = file.read()
except FileNotFoundError:
    print(f"ERROR: File '{INPUT_FILE}' not found!")
    exit(1)

# Find the summary table
table_marker = 'QUICK REFERENCE SUMMARY TABLE'
table_start = content.find(table_marker)

if table_start == -1:
    print("ERROR: Could not find summary table in the file")
    exit(1)

# Extract lines after the table marker
table_section = content[table_start:]
lines = table_section.split('\n')

# Find where data starts (after the header line)
data_lines = []
header_found = False

for line in lines:
    # Skip until we find the header
    if 'Feature' in line and 'Pearson' in line and not header_found:
        header_found = True
        continue
    
    # Start collecting data after header
    if header_found:
        # Stop at end markers
        if line.startswith('===') or 'END OF REPORT' in line:
            break
        
        # Skip empty lines
        if not line.strip():
            continue
        
        # Parse data line
        parts = line.split()
        if len(parts) >= 13 and parts[0].startswith('VB'):
            data_lines.append(parts)

print(f"✓ Found {len(data_lines)} features in the report")

if len(data_lines) == 0:
    print("ERROR: No data lines found. Please check the TXT file format.")
    exit(1)

# Create DataFrame
df = pd.DataFrame(data_lines, columns=[
    'Feature', 'Total_Rows', 'Valid_Points', 'NaN_Count', 'Valid_Pct',
    'Pearson', 'Spearman', 'Kendall', 'Mutual_Info', 'Best_Lag', 
    'Best_Lag_Corr', 'Granger_Price', 'Abs_Pearson'
])

# Convert numeric columns
numeric_cols = {
    'Total_Rows': int, 'Valid_Points': int, 'NaN_Count': int,
    'Valid_Pct': float, 'Pearson': float, 'Spearman': float, 
    'Kendall': float, 'Mutual_Info': float, 'Best_Lag': int,
    'Best_Lag_Corr': float, 'Abs_Pearson': float
}

for col, dtype in numeric_cols.items():
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if dtype == int:
            df[col] = df[col].fillna(0).astype(int)
    except:
        print(f"Warning: Could not convert column {col}")

# Extract feature groups and time periods
df['Feature_Group'] = df['Feature'].str.extract(r'(VB\d+)')[0]
df['Time_Period'] = df['Feature'].str.extract(r'T(\d+)')[0].astype(int)

print(f"✓ Parsed data successfully")
print(f"  Feature Groups: {sorted(df['Feature_Group'].unique())}")
print(f"  Time Periods: T{df['Time_Period'].min()} to T{df['Time_Period'].max()}")

# ============================================================================
# PLOT 1: CORRELATION HEATMAP
# ============================================================================
print("\n[1/4] Creating Correlation Heatmap...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('VB Feature Correlations: Multi-Metric Heatmap Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

metrics = ['Pearson', 'Spearman', 'Kendall', 'Mutual_Info', 'Best_Lag_Corr']
titles = ['Pearson Correlation', 'Spearman Correlation', 'Kendall Tau', 
          'Mutual Information', 'Best Lagged Correlation']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    pivot = df.pivot_table(values=metric, index='Time_Period', 
                           columns='Feature_Group', aggfunc='mean')
    
    cmap = 'YlOrRd' if metric == 'Mutual_Info' else 'RdBu_r'
    center = None if metric == 'Mutual_Info' else 0
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, center=center,
                cbar_kws={'label': metric}, ax=ax, linewidths=0.5)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('Feature Group', fontweight='bold')
    ax.set_ylabel('Time Period', fontweight='bold')

# Granger Causality heatmap
ax = axes[1, 2]
granger_pivot = df.pivot_table(
    values='Granger_Price', index='Time_Period', columns='Feature_Group',
    aggfunc=lambda x: (x == 'YES').sum()
)

sns.heatmap(granger_pivot, annot=True, fmt='g', cmap='Greens', 
            cbar_kws={'label': 'Granger Causality Count'}, 
            ax=ax, linewidths=0.5)
ax.set_title('Granger Causality (Count)', fontweight='bold', fontsize=12)
ax.set_xlabel('Feature Group', fontweight='bold')
ax.set_ylabel('Time Period', fontweight='bold')

plt.tight_layout()
output_file = f"{OUTPUT_DIR}01_correlation_heatmap.png"
plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ============================================================================
# PLOT 2: TIME PERIOD EVOLUTION
# ============================================================================
print("[2/4] Creating Time Period Evolution Plot...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('VB Feature Correlation Evolution Across Time Periods', 
             fontsize=16, fontweight='bold', y=0.995)

# Pearson correlation trends
ax = axes[0, 0]
for group in sorted(df['Feature_Group'].unique()):
    group_data = df[df['Feature_Group'] == group].sort_values('Time_Period')
    ax.plot(group_data['Time_Period'], group_data['Pearson'], 
            marker='o', label=group, linewidth=2, markersize=8)

ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Period', fontweight='bold', fontsize=11)
ax.set_ylabel('Pearson Correlation', fontweight='bold', fontsize=11)
ax.set_title('Pearson Correlation by Time Period', fontweight='bold', fontsize=12)
ax.legend(title='Feature Group', fontsize=9)
ax.grid(True, alpha=0.3)

# Absolute correlation strength
ax = axes[0, 1]
for group in sorted(df['Feature_Group'].unique()):
    group_data = df[df['Feature_Group'] == group].sort_values('Time_Period')
    ax.plot(group_data['Time_Period'], group_data['Abs_Pearson'], 
            marker='s', label=group, linewidth=2, markersize=8)

ax.set_xlabel('Time Period', fontweight='bold', fontsize=11)
ax.set_ylabel('Absolute Pearson Correlation', fontweight='bold', fontsize=11)
ax.set_title('Correlation Strength (Absolute)', fontweight='bold', fontsize=12)
ax.legend(title='Feature Group', fontsize=9)
ax.grid(True, alpha=0.3)

# Mutual Information trends
ax = axes[1, 0]
for group in sorted(df['Feature_Group'].unique()):
    group_data = df[df['Feature_Group'] == group].sort_values('Time_Period')
    ax.plot(group_data['Time_Period'], group_data['Mutual_Info'], 
            marker='^', label=group, linewidth=2, markersize=8)

ax.set_xlabel('Time Period', fontweight='bold', fontsize=11)
ax.set_ylabel('Mutual Information', fontweight='bold', fontsize=11)
ax.set_title('Mutual Information by Time Period', fontweight='bold', fontsize=12)
ax.legend(title='Feature Group', fontsize=9)
ax.grid(True, alpha=0.3)

# Data quality
ax = axes[1, 1]
for group in sorted(df['Feature_Group'].unique()):
    group_data = df[df['Feature_Group'] == group].sort_values('Time_Period')
    ax.plot(group_data['Time_Period'], group_data['Valid_Pct'], 
            marker='d', label=group, linewidth=2, markersize=8)

ax.set_xlabel('Time Period', fontweight='bold', fontsize=11)
ax.set_ylabel('Valid Data (%)', fontweight='bold', fontsize=11)
ax.set_title('Data Quality by Time Period', fontweight='bold', fontsize=12)
ax.legend(title='Feature Group', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = f"{OUTPUT_DIR}02_time_period_evolution.png"
plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ============================================================================
# PLOT 3: LAG ANALYSIS
# ============================================================================
print("[3/4] Creating Lag Analysis Plot...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('VB Feature Lag Analysis and Granger Causality', 
             fontsize=16, fontweight='bold', y=0.995)

# Best Lag Distribution
ax1 = fig.add_subplot(gs[0, :2])
lag_data = [df[df['Feature_Group'] == g]['Best_Lag'].values 
            for g in sorted(df['Feature_Group'].unique())]
labels = sorted(df['Feature_Group'].unique())

bp = ax1.boxplot(lag_data, labels=labels, patch_artist=True, 
                 showmeans=True, meanline=True)

colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Zero Lag')
ax1.set_xlabel('Feature Group', fontweight='bold', fontsize=11)
ax1.set_ylabel('Best Lag (periods)', fontweight='bold', fontsize=11)
ax1.set_title('Best Lag Distribution by Feature Group', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Lag vs Correlation
ax2 = fig.add_subplot(gs[0, 2])
color_map = dict(zip(sorted(df['Feature_Group'].unique()), colors))

for group in sorted(df['Feature_Group'].unique()):
    group_data = df[df['Feature_Group'] == group]
    ax2.scatter(group_data['Best_Lag'], group_data['Abs_Pearson'], 
                alpha=0.6, s=100, label=group, color=color_map[group])

ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax2.set_xlabel('Best Lag (periods)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Absolute Correlation', fontweight='bold', fontsize=11)
ax2.set_title('Lag vs Correlation Strength', fontweight='bold', fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Granger by Group
ax3 = fig.add_subplot(gs[1, 0])
granger_counts = df.groupby('Feature_Group')['Granger_Price'].apply(
    lambda x: (x == 'YES').sum()
).sort_values(ascending=False)

bars = ax3.bar(range(len(granger_counts)), granger_counts.values, 
               color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(granger_counts))))
ax3.set_xticks(range(len(granger_counts)))
ax3.set_xticklabels(granger_counts.index)
ax3.set_xlabel('Feature Group', fontweight='bold', fontsize=11)
ax3.set_ylabel('Granger Causality Count', fontweight='bold', fontsize=11)
ax3.set_title('Granger Causality by Group', fontweight='bold', fontsize=12)

for i, v in enumerate(granger_counts.values):
    ax3.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Granger by Time
ax4 = fig.add_subplot(gs[1, 1])
granger_time = df.groupby('Time_Period')['Granger_Price'].apply(
    lambda x: (x == 'YES').sum()
).sort_index()

ax4.plot(granger_time.index, granger_time.values, marker='o', 
         linewidth=2, markersize=10, color='darkgreen')
ax4.fill_between(granger_time.index, granger_time.values, alpha=0.3, color='lightgreen')
ax4.set_xlabel('Time Period', fontweight='bold', fontsize=11)
ax4.set_ylabel('Granger Causality Count', fontweight='bold', fontsize=11)
ax4.set_title('Granger Causality by Time Period', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

# Lag Improvement Heatmap
ax5 = fig.add_subplot(gs[1, 2])
df['Lag_Improvement'] = df['Best_Lag_Corr'].abs() - df['Abs_Pearson']

lag_imp_pivot = df.pivot_table(values='Lag_Improvement', 
                                index='Time_Period', 
                                columns='Feature_Group',
                                aggfunc='mean')

sns.heatmap(lag_imp_pivot, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Improvement'}, ax=ax5, linewidths=0.5)
ax5.set_title('Lagged Corr Improvement\n(Best Lag - Zero Lag)', 
              fontweight='bold', fontsize=12)
ax5.set_xlabel('Feature Group', fontweight='bold')
ax5.set_ylabel('Time Period', fontweight='bold')

# Pearson vs Best Lag Correlation
ax6 = fig.add_subplot(gs[2, :])
for group in sorted(df['Feature_Group'].unique()):
    group_data = df[df['Feature_Group'] == group]
    ax6.scatter(group_data['Pearson'], group_data['Best_Lag_Corr'], 
                alpha=0.6, s=120, label=group, color=color_map[group])

lims = [min(ax6.get_xlim()[0], ax6.get_ylim()[0]),
        max(ax6.get_xlim()[1], ax6.get_ylim()[1])]
ax6.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=2, label='No Improvement')

ax6.set_xlabel('Pearson (Zero Lag)', fontweight='bold', fontsize=11)
ax6.set_ylabel('Best Lagged Correlation', fontweight='bold', fontsize=11)
ax6.set_title('Zero Lag vs Best Lagged Correlation', fontweight='bold', fontsize=12)
ax6.legend(fontsize=9, ncol=7, loc='upper left')
ax6.grid(True, alpha=0.3)

plt.savefig(f"{OUTPUT_DIR}03_lag_analysis.png", dpi=FIGURE_DPI, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}03_lag_analysis.png")
plt.close()

# ============================================================================
# PLOT 4: CORRELATION COMPARISON
# ============================================================================
print("[4/4] Creating Correlation Comparison Plot...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('VB Feature Correlation Methods Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

# Pearson vs Spearman
ax = axes[0, 0]
for group in sorted(df['Feature_Group'].unique()):
    group_data = df[df['Feature_Group'] == group]
    ax.scatter(group_data['Pearson'], group_data['Spearman'], 
               alpha=0.6, s=80, label=group)

lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.3)
ax.set_xlabel('Pearson', fontweight='bold')
ax.set_ylabel('Spearman', fontweight='bold')
ax.set_title('Pearson vs Spearman', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Pearson vs Kendall
ax = axes[0, 1]
for group in sorted(df['Feature_Group'].unique()):
    group_data = df[df['Feature_Group'] == group]
    ax.scatter(group_data['Pearson'], group_data['Kendall'], 
               alpha=0.6, s=80, label=group)

lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.3)
ax.set_xlabel('Pearson', fontweight='bold')
ax.set_ylabel('Kendall', fontweight='bold')
ax.set_title('Pearson vs Kendall', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# MI vs Pearson
ax = axes[0, 2]
for group in sorted(df['Feature_Group'].unique()):
    group_data = df[df['Feature_Group'] == group]
    ax.scatter(group_data['Abs_Pearson'], group_data['Mutual_Info'], 
               alpha=0.6, s=80, label=group)

ax.set_xlabel('Abs Pearson', fontweight='bold')
ax.set_ylabel('Mutual Information', fontweight='bold')
ax.set_title('Linear vs Non-Linear', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Distribution by Group
ax = axes[1, 0]
pearson_data = [df[df['Feature_Group'] == g]['Pearson'].values 
                for g in sorted(df['Feature_Group'].unique())]
bp = ax.boxplot(pearson_data, labels=sorted(df['Feature_Group'].unique()), 
                patch_artist=True, showmeans=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Feature Group', fontweight='bold')
ax.set_ylabel('Pearson Correlation', fontweight='bold')
ax.set_title('Correlation Distribution', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Top Features
ax = axes[1, 1]
top_features = df.nlargest(15, 'Abs_Pearson')[['Feature', 'Abs_Pearson', 'Granger_Price']]
colors_granger = ['green' if x == 'YES' else 'orange' for x in top_features['Granger_Price']]

ax.barh(range(len(top_features)), top_features['Abs_Pearson'].values, color=colors_granger)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values, fontsize=9)
ax.set_xlabel('Abs Pearson', fontweight='bold')
ax.set_title('Top 15 Features', fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='With Granger'),
                   Patch(facecolor='orange', label='No Granger')]
ax.legend(handles=legend_elements, fontsize=9)

# Group Summary
ax = axes[1, 2]
summary_stats = df.groupby('Feature_Group').agg({
    'Abs_Pearson': 'mean',
    'Spearman': lambda x: x.abs().mean(),
    'Mutual_Info': 'mean'
})

x = np.arange(len(summary_stats))
width = 0.25

ax.bar(x - width, summary_stats['Abs_Pearson'], width, 
       label='Abs Pearson', alpha=0.8)
ax.bar(x, summary_stats['Spearman'], width, 
       label='Abs Spearman', alpha=0.8)
ax.bar(x + width, summary_stats['Mutual_Info'], width, 
       label='Mutual Info', alpha=0.8)

ax.set_xlabel('Feature Group', fontweight='bold')
ax.set_ylabel('Average Value', fontweight='bold')
ax.set_title('Average Metrics by Group', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(summary_stats.index)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}04_correlation_comparison.png", dpi=FIGURE_DPI, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}04_correlation_comparison.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)

print("\nTop 10 Features by Absolute Pearson:")
top10 = df.nlargest(10, 'Abs_Pearson')[['Feature', 'Pearson', 'Spearman', 
                                         'Mutual_Info', 'Granger_Price']]
print(top10.to_string(index=False))

granger_yes = df[df['Granger_Price'] == 'YES']
print(f"\nFeatures with Granger Causality: {len(granger_yes)}/{len(df)} ({100*len(granger_yes)/len(df):.1f}%)")

print("\nGroup Statistics:")
group_stats = df.groupby('Feature_Group').agg({
    'Pearson': lambda x: f"{x.mean():.4f}",
    'Abs_Pearson': lambda x: f"{x.mean():.4f}",
    'Mutual_Info': lambda x: f"{x.mean():.4f}",
    'Granger_Price': lambda x: f"{(x == 'YES').sum()}/{len(x)}"
})
group_stats.columns = ['Avg Pearson', 'Avg Abs Pearson', 'Avg MI', 'Granger']
print(group_stats.to_string())

print("\n" + "="*100)
print("✓ VISUALIZATION COMPLETE!")
print("="*100)
print(f"All plots saved in: {OUTPUT_DIR}")
print("  - 01_correlation_heatmap.png")
print("  - 02_time_period_evolution.png")
print("  - 03_lag_analysis.png")
print("  - 04_correlation_comparison.png")
print("="*100)