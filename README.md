# CDX Tranche Pricing - Copula Models Comparison

Comparison of Gaussian and G-VG copula models for pricing CDX.NA.IG.45 tranches.

## Overview

This project implements and compares two copula models for credit default swap index tranche pricing:
1. **Gaussian Copula**: Li (2000) one-factor model with single correlation parameter
2. **G-VG Copula**: Mixed Gaussian-Variance Gamma copula with stochastic correlation

## Data

Market data as of October 23, 2025:
- 125 constituent CDS spreads (CDX.NA.IG.45)
- USD OIS discount curve
- Market tranche prices for 5-year maturity

All data files are located in `data/` directory.

## Project Structure

```
.
├── notebooks/
│   ├── gaussian_copula_pricing.ipynb    # Gaussian copula implementation
│   └── gvg_copula_pricing.ipynb         # G-VG copula implementation
├── data/
│   ├── cdx_constituents.csv             # CDS spreads (125 companies)
│   ├── ois_curve.csv                    # Interest rate curve
│   └── cdx_market_data.json             # Market tranche prices
├── results/
│   ├── results.json                     # Gaussian model results
│   ├── results_gvg.json                 # G-VG model results
│   ├── pricing_comparison.csv           # Gaussian pricing comparison
│   ├── pricing_comparison_gvg.csv       # G-VG pricing comparison
│   └── model_comparison.csv             # Model comparison summary
└── README.md
```

## Usage

```bash
# Run Gaussian Copula model
jupyter notebook notebooks/gaussian_copula_pricing.ipynb

# Run G-VG Copula model
jupyter notebook notebooks/gvg_copula_pricing.ipynb
```

## Results

### Model Comparison

| Model | APE (bps) | MAE (bps) | Correlation | Correlation Skew |
|-------|-----------|-----------|-------------|------------------|
| Gaussian | 431.03 | 86.21 | ρ = 0.337 | 0.00pp |
| G-VG | 437.58 | 87.52 | ρ = 0.25-0.60 | 35.00pp |

### Gaussian Copula
- **Calibrated correlation (ρ)**: 0.337 (33.67%)
- **Absolute Pricing Error (APE)**: 431.03 bps
- **Mean Absolute Error (MAE)**: 86.21 bps
- Uses single flat correlation parameter

### G-VG Copula
- **Low correlation (ρ_low)**: 0.25 (25%)
- **High correlation (ρ_high)**: 0.60 (60%)
- **Regime probability**: 0.70
- **Degrees of freedom**: 8
- **APE**: 437.58 bps
- **MAE**: 87.52 bps
- Captures correlation skew via regime-switching framework

## Key Findings

1. **Gaussian copula** performs slightly better in terms of APE (431 vs 438 bps)
2. **G-VG copula** introduces correlation skew (25%-60%) but doesn't improve pricing accuracy
3. Market correlation skew: 47.7% → 67.6% (implied base correlations)
4. Both models struggle to price mezzanine tranches accurately

The results suggest that while the G-VG model adds complexity through regime-switching and fat-tailed distributions, it does not necessarily improve pricing accuracy for this specific dataset. The Gaussian copula's simpler structure achieves comparable (slightly better) performance.

## Implementation Details

Both implementations include:
- Survival probability bootstrapping from CDS spreads
- Large Homogeneous Portfolio (LHP) approximation
- Grid-based numerical integration for loss distribution
- Calibration via grid search (G-VG) or scalar optimization (Gaussian)

## References

- Li, D. X. (2000). "On Default Correlation: A Copula Function Approach"
- Hull, J., & White, A. (2004). "Valuation of a CDO and nth to Default CDS"
- Madan, D. B., & Seneta, E. (1990). "The Variance Gamma (V.G.) Model for Share Market Returns"
