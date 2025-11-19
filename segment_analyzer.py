#!/usr/bin/env python3
"""
Price Segmentation Analyzer

Analyzes optimal price segmentation strategies for property price prediction.
Tests different segmentation approaches and recommends the best configuration.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 80)
print("PRICE SEGMENTATION ANALYZER")
print("=" * 80)

# Load data
print("\n### Loading data...")
df = pd.read_csv('/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/properati_preprocessed_20251016_204913.csv')

# Apply same filters as train_simple_model.py
df = df[df['currency'] == 'USD'].copy()
df = df[(df['price'] >= 30_000) & (df['price'] <= 600_000)].copy()

# Filter by price per sqm
price_per_sqm = df['price'] / df['area']
df = df[(price_per_sqm >= 800) & (price_per_sqm <= 5000)].copy()

# Remove missing coordinates
df = df.dropna(subset=['latitude', 'longitude'])

print(f"Filtered dataset: {len(df):,} properties")
print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"Mean price: ${df['price'].mean():,.0f}")
print(f"Median price: ${df['price'].median():,.0f}")

# Define segmentation strategies to test
segmentation_strategies = {
    '2_segments': [
        {'name': 'low', 'min': 30_000, 'max': 150_000},
        {'name': 'high', 'min': 150_000, 'max': 600_000}
    ],
    '3_segments_v1': [
        {'name': 'budget', 'min': 30_000, 'max': 120_000},
        {'name': 'mid', 'min': 120_000, 'max': 300_000},
        {'name': 'premium', 'min': 300_000, 'max': 600_000}
    ],
    '3_segments_v2': [
        {'name': 'budget', 'min': 30_000, 'max': 100_000},
        {'name': 'mid', 'min': 100_000, 'max': 250_000},
        {'name': 'premium', 'min': 250_000, 'max': 600_000}
    ],
    '4_segments': [
        {'name': 'budget', 'min': 30_000, 'max': 100_000},
        {'name': 'mid_low', 'min': 100_000, 'max': 200_000},
        {'name': 'mid_high', 'min': 200_000, 'max': 400_000},
        {'name': 'premium', 'min': 400_000, 'max': 600_000}
    ]
}

print("\n" + "=" * 80)
print("ANALYZING SEGMENTATION STRATEGIES")
print("=" * 80)

results = []

for strategy_name, segments in segmentation_strategies.items():
    print(f"\n### Strategy: {strategy_name}")
    print("-" * 80)

    segment_stats = []

    for seg in segments:
        mask = (df['price'] >= seg['min']) & (df['price'] < seg['max'])
        seg_data = df[mask]

        count = len(seg_data)
        pct = (count / len(df)) * 100
        mean_price = seg_data['price'].mean()
        std_price = seg_data['price'].std()
        cv = (std_price / mean_price) * 100  # Coefficient of variation

        segment_stats.append({
            'segment': seg['name'],
            'min': seg['min'],
            'max': seg['max'],
            'count': count,
            'pct': pct,
            'mean': mean_price,
            'std': std_price,
            'cv': cv
        })

        print(f"{seg['name']:15s}: ${seg['min']:>7,} - ${seg['max']:>7,} | "
              f"{count:>6,} props ({pct:>5.1f}%) | "
              f"Mean: ${mean_price:>7,.0f} | "
              f"Std: ${std_price:>6,.0f} | "
              f"CV: {cv:>5.1f}%")

    # Calculate balance score (how evenly distributed)
    counts = [s['count'] for s in segment_stats]
    balance_score = min(counts) / max(counts) if counts else 0

    # Calculate average coefficient of variation (lower = more homogeneous segments)
    avg_cv = np.mean([s['cv'] for s in segment_stats])

    # Calculate weighted CV (prefer strategies with low CV in high-count segments)
    weighted_cv = sum(s['cv'] * s['pct'] for s in segment_stats) / 100

    results.append({
        'strategy': strategy_name,
        'n_segments': len(segments),
        'balance_score': balance_score,
        'avg_cv': avg_cv,
        'weighted_cv': weighted_cv,
        'stats': segment_stats
    })

    print(f"\nBalance Score: {balance_score:.3f} (1.0 = perfect balance)")
    print(f"Avg CV: {avg_cv:.1f}% (lower = more homogeneous)")
    print(f"Weighted CV: {weighted_cv:.1f}%")

# Find best strategy
print("\n" + "=" * 80)
print("STRATEGY COMPARISON")
print("=" * 80)

results_df = pd.DataFrame([{
    'Strategy': r['strategy'],
    'Segments': r['n_segments'],
    'Balance': f"{r['balance_score']:.3f}",
    'Avg CV': f"{r['avg_cv']:.1f}%",
    'Weighted CV': f"{r['weighted_cv']:.1f}%"
} for r in results])

print("\n" + results_df.to_string(index=False))

# Recommendation logic
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Score each strategy
for r in results:
    # Lower CV is better (more homogeneous segments)
    # Higher balance is better (even distribution)
    # Penalize too many segments (overfitting risk)
    cv_score = 100 - r['weighted_cv']  # Lower CV = higher score
    balance_score = r['balance_score'] * 100
    segment_penalty = (5 - r['n_segments']) * 10  # Prefer 3-4 segments

    r['total_score'] = cv_score + balance_score + segment_penalty

best_strategy = max(results, key=lambda x: x['total_score'])

print(f"\n🏆 Recommended Strategy: {best_strategy['strategy']}")
print(f"   Number of segments: {best_strategy['n_segments']}")
print(f"   Balance score: {best_strategy['balance_score']:.3f}")
print(f"   Weighted CV: {best_strategy['weighted_cv']:.1f}%")
print(f"\nSegment Details:")

for stat in best_strategy['stats']:
    print(f"  {stat['segment']:15s}: ${stat['min']:>7,} - ${stat['max']:>7,} "
          f"({stat['count']:>6,} properties, {stat['pct']:>5.1f}%)")

# Create visualization
print("\n### Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Price distribution with segment boundaries
ax1 = axes[0, 0]
ax1.hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
for stat in best_strategy['stats']:
    ax1.axvline(stat['min'], color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(stat['min'], ax1.get_ylim()[1] * 0.9, stat['segment'],
             rotation=90, verticalalignment='top', fontsize=10, color='red')
ax1.set_xlabel('Price ($)')
ax1.set_ylabel('Count')
ax1.set_title(f'Price Distribution with {best_strategy["strategy"]} Boundaries')
ax1.grid(True, alpha=0.3)

# Plot 2: Properties per segment
ax2 = axes[0, 1]
segment_names = [s['segment'] for s in best_strategy['stats']]
segment_counts = [s['count'] for s in best_strategy['stats']]
bars = ax2.bar(segment_names, segment_counts, edgecolor='black')
for i, (bar, count) in enumerate(zip(bars, segment_counts)):
    height = bar.get_height()
    pct = best_strategy['stats'][i]['pct']
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=10)
ax2.set_ylabel('Number of Properties')
ax2.set_title('Properties per Segment')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Coefficient of Variation per segment
ax3 = axes[1, 0]
segment_cvs = [s['cv'] for s in best_strategy['stats']]
bars = ax3.bar(segment_names, segment_cvs, color='orange', edgecolor='black')
for bar, cv in zip(bars, segment_cvs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{cv:.1f}%',
             ha='center', va='bottom', fontsize=10)
ax3.set_ylabel('Coefficient of Variation (%)')
ax3.set_title('Price Homogeneity per Segment (Lower = Better)')
ax3.axhline(df['price'].std() / df['price'].mean() * 100,
            color='red', linestyle='--', label='Overall CV', linewidth=2)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Mean price and std per segment
ax4 = axes[1, 1]
x = np.arange(len(segment_names))
width = 0.35
means = [s['mean'] for s in best_strategy['stats']]
stds = [s['std'] for s in best_strategy['stats']]

bars1 = ax4.bar(x - width/2, means, width, label='Mean Price',
                color='skyblue', edgecolor='black')
bars2 = ax4.bar(x + width/2, stds, width, label='Std Dev',
                color='lightcoral', edgecolor='black')

ax4.set_ylabel('Price ($)')
ax4.set_title('Mean Price and Standard Deviation per Segment')
ax4.set_xticks(x)
ax4.set_xticklabels(segment_names)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Format y-axis as currency
for ax in [ax4]:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

plt.tight_layout()
output_path = 'data/segment_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to {output_path}")

# Save detailed analysis
analysis_path = 'data/segment_analysis_details.txt'
with open(analysis_path, 'w') as f:
    f.write("PRICE SEGMENTATION ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset: {len(df):,} properties\n")
    f.write(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}\n\n")

    f.write(f"RECOMMENDED STRATEGY: {best_strategy['strategy']}\n")
    f.write("-" * 80 + "\n\n")

    for stat in best_strategy['stats']:
        f.write(f"Segment: {stat['segment']}\n")
        f.write(f"  Range: ${stat['min']:,} - ${stat['max']:,}\n")
        f.write(f"  Properties: {stat['count']:,} ({stat['pct']:.1f}%)\n")
        f.write(f"  Mean: ${stat['mean']:,.0f}\n")
        f.write(f"  Std Dev: ${stat['std']:,.0f}\n")
        f.write(f"  CV: {stat['cv']:.1f}%\n\n")

    f.write(f"Balance Score: {best_strategy['balance_score']:.3f}\n")
    f.write(f"Weighted CV: {best_strategy['weighted_cv']:.1f}%\n")

print(f"✅ Detailed analysis saved to {analysis_path}")

# Create recommended config
print("\n### Creating recommended configuration...")
config = {
    'strategy_name': best_strategy['strategy'],
    'segments': []
}

for stat in best_strategy['stats']:
    config['segments'].append({
        'name': stat['segment'],
        'price_min': int(stat['min']),
        'price_max': int(stat['max']),
        'expected_count': int(stat['count']),
        'expected_pct': round(stat['pct'], 1)
    })

import json
config_path = 'segment_config_recommended.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Recommended config saved to {config_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nNext steps:")
print(f"1. Review visualization: {output_path}")
print(f"2. Review recommended config: {config_path}")
print(f"3. Use this config in train_segmented_model.py")
