#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Creating visualizations...")
gaussian_pricing = pd.read_csv('results/gaussian_multi_tenor_pricing.csv')
corr_term_struct = pd.read_csv('results/correlation_term_structure.csv')
summary = pd.read_csv('results/multi_tenor_summary.csv')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(corr_term_struct['Maturity_Years'], corr_term_struct['Correlation_Pct'],
        marker='o', linewidth=2, markersize=10, color='#2E86AB')

ax.set_xlabel('Maturity (Years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Asset Correlation (%)', fontsize=12, fontweight='bold')
ax.set_title('Correlation Term Structure\nGaussian Copula Model',
             fontsize=14, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3)
ax.set_xlim(0, 11)
ax.set_ylim(50, 70)

# Add value labels
for i, row in corr_term_struct.iterrows():
    ax.annotate(f'{row["Correlation_Pct"]:.1f}%',
                xy=(row['Maturity_Years'], row['Correlation_Pct']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/correlation_term_structure.png', dpi=300, bbox_inches='tight')
print("✓ Saved correlation_term_structure.png")
plt.close()


fig, ax = plt.subplots(figsize=(12, 6))

tenors = summary['Tenor'].tolist()
mae = summary['MAE_bps'].tolist()
rmse = summary['RMSE_bps'].tolist()

x = np.arange(len(tenors))
width = 0.35

bars1 = ax.bar(x - width/2, mae, width, label='MAE', color='#A23B72', alpha=0.8)
bars2 = ax.bar(x + width/2, rmse, width, label='RMSE', color='#F18F01', alpha=0.8)

ax.set_xlabel('Tenor', fontsize=12, fontweight='bold')
ax.set_ylabel('Error (bps)', fontsize=12, fontweight='bold')
ax.set_title('Pricing Errors by Tenor\nGaussian Copula Model',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(tenors)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/pricing_errors_by_tenor.png', dpi=300, bbox_inches='tight')
print("✓ Saved pricing_errors_by_tenor.png")
plt.close()


# Pivot data for heatmap
heatmap_data = gaussian_pricing.pivot(index='Tranche', columns='Tenor', values='Error_bps')

# Reorder rows for better visualization
tranche_order = ['Equity 0-3%', 'Mezz 3-7%', 'Mezz 7-10%', 'Senior 10-15%', 'Senior 15-100%']
heatmap_data = heatmap_data.reindex(tranche_order)

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
            cbar_kws={'label': 'Pricing Error (bps)'},
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title('Pricing Errors: Gaussian Copula Model\n(bps)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Tenor', fontsize=12, fontweight='bold')
ax.set_ylabel('Tranche', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/pricing_errors_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved pricing_errors_heatmap.png")
plt.close()


fig, ax = plt.subplots(figsize=(14, 7))

for tranche in tranche_order:
    tranche_data = gaussian_pricing[gaussian_pricing['Tranche'] == tranche]
    tenors_years = [float(t.replace('Y','')) for t in tranche_data['Tenor']]
    exp_loss = tranche_data['Expected_Loss_pct'].tolist()

    ax.plot(tenors_years, exp_loss, marker='o', linewidth=2, markersize=8, label=tranche)

ax.set_xlabel('Maturity (Years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Loss (%)', fontsize=12, fontweight='bold')
ax.set_title('Expected Tranche Losses Across Tenors\nGaussian Copula Model',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 11)

plt.tight_layout()
plt.savefig('results/expected_loss_by_tranche.png', dpi=300, bbox_inches='tight')
print("✓ Saved expected_loss_by_tranche.png")
plt.close()


tenor_5y = gaussian_pricing[gaussian_pricing['Tenor'] == '5Y']

fig, ax = plt.subplots(figsize=(12, 7))

tranches_short = ['0-3%', '3-7%', '7-10%', '10-15%', '15-100%']
x = np.arange(len(tranches_short))
width = 0.35

market = tenor_5y['Market_Spread_bps'].tolist()
model = tenor_5y['Model_Spread_bps'].tolist()

bars1 = ax.bar(x - width/2, market, width, label='Market', color='#06A77D', alpha=0.8)
bars2 = ax.bar(x + width/2, model, width, label='Model', color='#D5A021', alpha=0.8)

ax.set_xlabel('Tranche', fontsize=12, fontweight='bold')
ax.set_ylabel('Spread (bps)', fontsize=12, fontweight='bold')
ax.set_title('Market vs Model Spreads: 5Y Tenor\nGaussian Copula (ρ = 62.1%)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(tranches_short)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('results/market_vs_model_5y.png', dpi=300, bbox_inches='tight')
print("✓ Saved market_vs_model_5y.png")
plt.close()

print("\n✓ All visualizations created successfully!")
print("  Files saved in results/ directory")
