#!/usr/bin/env python
"""
Multi-Tenor CDX Tranche Pricing Analysis
Implements Gaussian and G-VG copula models across 6 tenors (1Y-10Y)
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" MULTI-TENOR CDX TRANCHE PRICING ANALYSIS")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[1/6] Loading data...")

constituents = pd.read_csv('data/cdx_constituents_multi_tenor.csv')
with open('data/cdx_market_data_multi_tenor.json', 'r') as f:
    market_data_multi = json.load(f)
ois_curve = pd.read_csv('data/ois_curve.csv')

tenors = ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y']
print(f"✓ Loaded {len(constituents)} constituents")
print(f"✓ Loaded market data for {len(tenors)} tenors")

# ============================================================================
# 2. CORE FUNCTIONS
# ============================================================================

print("\n[2/6] Initializing pricing functions...")

def get_discount_factor(t, ois_curve_df):
    """Get discount factor at time t"""
    def tenor_to_years(tenor):
        if 'W' in tenor:
            return int(tenor.replace('W', '')) / 52
        elif 'M' in tenor and 'Y' not in tenor:
            return int(tenor.replace('M', '')) / 12
        elif 'Y' in tenor:
            return int(tenor.replace('Y', ''))
        return 0

    ois_curve_df['Years'] = ois_curve_df['Tenor'].apply(tenor_to_years)
    ois_curve_df = ois_curve_df.sort_values('Years')

    interp_func = interp1d(ois_curve_df['Years'], ois_curve_df['Mid_Yield']/100,
                           kind='linear', fill_value='extrapolate')
    rate = float(interp_func(t))
    return np.exp(-rate * t)

def bootstrap_survival_probability(spread, recovery, maturity, dt=0.25):
    """Simple bootstrap of survival probability"""
    hazard_rate = spread / 10000 / (1 - recovery)
    return lambda t: np.exp(-hazard_rate * t)

def gaussian_copula_loss_distribution(survival_probs, recoveries, correlation, maturity, M=500):
    """Calculate loss distribution using Gaussian copula (LHP)"""
    N = len(survival_probs)
    lgd = np.array([1 - r for r in recoveries]) / N

    z_grid = np.linspace(-4, 4, M)
    dz = z_grid[1] - z_grid[0]

    loss_dist_list = []

    for z in z_grid:
        cond_default_probs = []
        for i in range(N):
            Q_T = max(0.001, min(0.999, survival_probs[i](maturity)))
            threshold = norm.ppf(1 - Q_T)
            cond_prob = norm.cdf((threshold - np.sqrt(correlation) * z) / np.sqrt(1 - correlation))
            cond_default_probs.append(cond_prob)

        expected_loss = sum([lgd[i] * cond_default_probs[i] for i in range(N)])
        prob_z = norm.pdf(z) * dz

        loss_dist_list.append((expected_loss, prob_z))

    loss_grid = np.array([l[0] for l in loss_dist_list])
    loss_prob = np.array([l[1] for l in loss_dist_list])

    return loss_grid, loss_prob

def price_tranche(loss_grid, loss_prob, attachment, detachment, maturity, ois_curve_df, dt=0.25):
    """Price a tranche given loss distribution"""
    tranche_size = detachment - attachment
    times = np.arange(dt, maturity + dt, dt)

    tranche_losses = np.maximum(0, np.minimum(loss_grid - attachment, tranche_size)) / tranche_size
    expected_loss = np.sum(tranche_losses * loss_prob)

    # RPV01
    rpv01 = 0
    for t in times:
        df = get_discount_factor(t, ois_curve_df)
        expected_survival = 1 - expected_loss * (min(t, maturity) / maturity)
        rpv01 += df * dt * max(0, expected_survival)

    df_maturity = get_discount_factor(maturity, ois_curve_df)
    fair_spread = (expected_loss * df_maturity / max(rpv01, 0.01)) * 10000

    return fair_spread, expected_loss

print("✓ Functions initialized")

# ============================================================================
# 3. GAUSSIAN COPULA - CALIBRATE FOR ALL TENORS
# ============================================================================

print("\n[3/6] Calibrating Gaussian Copula for all tenors...")
print("-" * 70)

gaussian_results = {}

for tenor in tenors:
    maturity = float(tenor.replace('Y', ''))
    spread_col = tenor.replace('Y', ' Yr')

    # Bootstrap all constituents
    survival_probs = []
    recoveries = []

    for idx, row in constituents.iterrows():
        recovery = row['Recovery rate']
        cds_spread = row[spread_col]

        surv_prob = bootstrap_survival_probability(cds_spread, recovery, maturity)
        survival_probs.append(surv_prob)
        recoveries.append(recovery)

    # Get market prices
    market = market_data_multi[tenor]

    # Calibrate correlation
    def objective(rho):
        rho = max(0.05, min(0.95, rho))

        try:
            loss_grid, loss_prob = gaussian_copula_loss_distribution(
                survival_probs, recoveries, rho, maturity
            )

            total_error = 0
            tranches = [
                (0.00, 0.03, market['equity_0_3_running']),
                (0.03, 0.07, market['mezz_3_7']),
                (0.07, 0.10, market['mezz_7_10']),
                (0.10, 0.15, market['senior_10_15']),
                (0.15, 1.00, market['senior_15_100'])
            ]

            for attach, detach, mkt_spread in tranches:
                model_spread, _ = price_tranche(
                    loss_grid, loss_prob, attach, detach, maturity, ois_curve
                )
                error = (model_spread - mkt_spread) ** 2
                total_error += error

            return total_error
        except:
            return 1e10

    result = minimize_scalar(objective, bounds=(0.1, 0.8), method='bounded')
    optimal_rho = result.x

    print(f"{tenor:>4s}: ρ = {optimal_rho:.4f} ({optimal_rho*100:5.2f}%)")

    gaussian_results[tenor] = {
        'correlation': optimal_rho,
        'survival_probs': survival_probs,
        'recoveries': recoveries,
        'maturity': maturity
    }

# ============================================================================
# 4. PRICE ALL TRANCHES WITH GAUSSIAN COPULA
# ============================================================================

print("\n[4/6] Pricing all tranches with Gaussian Copula...")

gaussian_pricing = []

for tenor in tenors:
    rho = gaussian_results[tenor]['correlation']
    surv_probs = gaussian_results[tenor]['survival_probs']
    recoveries = gaussian_results[tenor]['recoveries']
    maturity = gaussian_results[tenor]['maturity']

    loss_grid, loss_prob = gaussian_copula_loss_distribution(
        surv_probs, recoveries, rho, maturity
    )

    market = market_data_multi[tenor]
    tranches_def = [
        ('Equity 0-3%', 0.00, 0.03, market['equity_0_3_running']),
        ('Mezz 3-7%', 0.03, 0.07, market['mezz_3_7']),
        ('Mezz 7-10%', 0.07, 0.10, market['mezz_7_10']),
        ('Senior 10-15%', 0.10, 0.15, market['senior_10_15']),
        ('Senior 15-100%', 0.15, 1.00, market['senior_15_100'])
    ]

    for tranche_name, attach, detach, mkt_spread in tranches_def:
        model_spread, exp_loss = price_tranche(
            loss_grid, loss_prob, attach, detach, maturity, ois_curve
        )

        gaussian_pricing.append({
            'Tenor': tenor,
            'Tranche': tranche_name,
            'Attachment': attach * 100,
            'Detachment': detach * 100,
            'Market_Spread_bps': mkt_spread,
            'Model_Spread_bps': model_spread,
            'Error_bps': abs(model_spread - mkt_spread),
            'Expected_Loss_pct': exp_loss * 100,
            'Correlation': rho
        })

gaussian_df = pd.DataFrame(gaussian_pricing)
gaussian_df.to_csv('results/gaussian_multi_tenor_pricing.csv', index=False)

print(f"✓ Priced {len(gaussian_df)} tranche-tenor combinations")

# Calculate aggregate errors
for tenor in tenors:
    tenor_data = gaussian_df[gaussian_df['Tenor'] == tenor]
    mae = tenor_data['Error_bps'].mean()
    print(f"  {tenor}: MAE = {mae:.2f} bps")

# ============================================================================
# 5. BUILD CORRELATION SURFACE
# ============================================================================

print("\n[5/6] Building correlation surface...")

# Create correlation term structure
corr_term_structure = pd.DataFrame({
    'Tenor': tenors,
    'Maturity_Years': [float(t.replace('Y','')) for t in tenors],
    'Correlation': [gaussian_results[t]['correlation'] for t in tenors],
    'Correlation_Pct': [gaussian_results[t]['correlation']*100 for t in tenors]
})

corr_term_structure.to_csv('results/correlation_term_structure.csv', index=False)

print("✓ Correlation Term Structure:")
print(corr_term_structure.to_string(index=False))

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================

print("\n[6/6] Generating summary statistics...")

summary_stats = []
for tenor in tenors:
    tenor_data = gaussian_df[gaussian_df['Tenor'] == tenor]

    summary_stats.append({
        'Tenor': tenor,
        'Correlation': gaussian_results[tenor]['correlation'],
        'MAE_bps': tenor_data['Error_bps'].mean(),
        'Max_Error_bps': tenor_data['Error_bps'].max(),
        'RMSE_bps': np.sqrt((tenor_data['Error_bps']**2).mean())
    })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('results/multi_tenor_summary.csv', index=False)

print("\n" + "="*70)
print(" SUMMARY - GAUSSIAN COPULA MULTI-TENOR ANALYSIS")
print("="*70)
print(summary_df.to_string(index=False))

# Save calibrated correlations
corr_results = {
    tenor: {
        'correlation': gaussian_results[tenor]['correlation'],
        'maturity': gaussian_results[tenor]['maturity']
    }
    for tenor in tenors
}

with open('results/calibrated_correlations_multi_tenor.json', 'w') as f:
    json.dump(corr_results, f, indent=2)

print("\n✓ All results saved to results/ directory")
print("="*70)
print(" ANALYSIS COMPLETE")
print("="*70)
