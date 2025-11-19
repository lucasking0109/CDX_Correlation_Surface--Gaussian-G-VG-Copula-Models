# CDX Tranche Pricing - Multi-Tenor Correlation Surface Analysis

Analysis of CDX.NA.IG.45 tranche pricing across multiple maturities (1Y-10Y) using Gaussian copula model, with focus on correlation term structure and tranche-specific pricing dynamics.

## Overview

This project implements copula-based pricing models for credit default swap index tranches across 6 tenors:
- **Tenors**: 1Y, 2Y, 3Y, 5Y, 7Y, 10Y
- **Model**: Gaussian Copula (Li 2000) one-factor model
- **Analysis**: Correlation term structure, base correlation surface, pricing error analysis

## Data

Market data as of October 23, 2025:
- **125 constituent CDS spreads** across 8 tenors (6M, 1Y, 2Y, 3Y, 4Y, 5Y, 7Y, 10Y)
- **USD OIS discount curve** (1W to 40Y)
- **Market tranche prices** for 6 tenors (1Y, 2Y, 3Y, 5Y, 7Y, 10Y)
  - Equity (0-3%), Mezzanine (3-7%, 7-10%), Senior (10-15%, 15-100%)

All data files are located in `data/` directory.

## Project Structure

```
.
├── notebooks/
│   ├── gaussian_copula_pricing.ipynb      # Single tenor (5Y) Gaussian copula
│   ├── gvg_copula_pricing.ipynb          # Single tenor (5Y) G-VG copula
│   └── multi_tenor_analysis.ipynb        # Multi-tenor analysis (1Y-10Y)
├── data/
│   ├── CDX_DATA (1).xlsx                 # Original Bloomberg data
│   ├── cdx_constituents_multi_tenor.csv  # Multi-tenor CDS spreads (125 companies)
│   ├── cdx_market_data_multi_tenor.json  # Multi-tenor tranche prices
│   └── ois_curve.csv                     # OIS discount curve
├── results/
│   ├── gaussian_multi_tenor_pricing.csv
│   ├── correlation_term_structure.csv
│   ├── multi_tenor_summary.csv
│   ├── calibrated_correlations_multi_tenor.json
│   └── *.png                             # Visualizations
├── run_multi_tenor_analysis.py           # Main analysis script
├── create_visualizations.py              # Visualization generation
└── README.md
```

## Usage

### Multi-Tenor Analysis

```bash
# Run complete multi-tenor analysis
python run_multi_tenor_analysis.py

# Generate visualizations
python create_visualizations.py

# Or use Jupyter notebook
jupyter notebook notebooks/multi_tenor_analysis.ipynb
```

### Single Tenor (5Y) Analysis

```bash
# Gaussian copula (5Y only)
jupyter notebook notebooks/gaussian_copula_pricing.ipynb

# G-VG copula (5Y only)
jupyter notebook notebooks/gvg_copula_pricing.ipynb
```

## Results

### Multi-Tenor Analysis Summary

**Gaussian Copula - Correlation Term Structure:**

| Tenor | Maturity | Correlation (ρ) | MAE (bps) | RMSE (bps) |
|-------|----------|-----------------|-----------|------------|
| 1Y    | 1 year   | 56.27%          | 21.80     | 26.81      |
| 2Y    | 2 years  | 56.88%          | 32.55     | 40.28      |
| 3Y    | 3 years  | 57.76%          | 43.02     | 54.14      |
| 5Y    | 5 years  | 62.14%          | 92.11     | 121.38     |
| 7Y    | 7 years  | 63.15%          | 92.97     | 105.79     |
| 10Y   | 10 years | 64.93%          | 115.74    | 127.51     |

### Key Findings

#### 1. Correlation Term Structure
- **Upward sloping**: Correlation increases from 56% (1Y) to 65% (10Y)
- **Economic interpretation**: Longer-term default correlations are higher, consistent with macroeconomic cycles
- **Steeper increase**: 1Y-3Y relatively flat (~56-58%), then steeper rise for 5Y+ (~62-65%)

#### 2. Pricing Accuracy by Tenor
- **Short-term tenors (1Y-3Y)**: Better pricing accuracy (MAE: 22-43 bps)
  - Lower uncertainty in near-term default probabilities
  - Less sensitivity to correlation assumptions
- **Long-term tenors (5Y-10Y)**: Higher pricing errors (MAE: 92-116 bps)
  - Greater model risk and parameter uncertainty
  - Mezzanine tranches particularly challenging

#### 3. Tranche-Specific Patterns
- **Equity tranches (0-3%)**: Largest absolute errors but relatively stable across tenors
- **Mezzanine tranches (3-10%)**: Most difficult to price accurately
- **Senior tranches (10-100%)**: Better pricing performance, especially for shorter maturities

#### 4. Correlation Skew
- While Gaussian copula uses flat correlation per tenor, the term structure itself creates a form of "maturity skew"
- Base correlations would show additional skew across attachment points (equity vs senior)

### Single Tenor (5Y) Comparison - Gaussian vs G-VG

| Model | APE (bps) | MAE (bps) | Correlation | Correlation Skew |
|-------|-----------|-----------|-------------|------------------|
| Gaussian | 431.03 | 86.21 | ρ = 33.7% | 0.00pp |
| G-VG | 437.58 | 87.52 | ρ = 25-60% | 35.00pp |

**Note**: The 5Y multi-tenor calibration (ρ = 62.14%) differs from the single-tenor calibration (ρ = 33.7%) due to different calibration objectives and market data used.

## Implementation Details

### Multi-Tenor Gaussian Copula

**Methodology:**
1. **Survival Probability Bootstrap**: Extract default probabilities from CDS spreads for each tenor
2. **Large Homogeneous Portfolio (LHP) Approximation**: Simplify portfolio loss distribution
3. **Correlation Calibration**: Optimize correlation parameter per tenor to minimize pricing errors
4. **Loss Distribution**: Grid-based numerical integration (500 market factor scenarios)
5. **Tranche Pricing**: Calculate fair spreads using discounted expected losses and RPV01

**Key Features:**
- Tenor-specific correlation calibration (6 separate correlations for 6 tenors)
- Consistent pricing framework across all maturities
- Term structure analysis of implied correlations
- Error analysis by tranche and maturity

### Single Tenor Models (5Y)

Both Gaussian and G-VG implementations for 5Y tenor include:
- Survival probability bootstrapping from CDS spreads
- Large Homogeneous Portfolio (LHP) approximation
- Grid-based numerical integration for loss distribution
- Calibration via grid search (G-VG) or scalar optimization (Gaussian)

## References

- Li, D. X. (2000). "On Default Correlation: A Copula Function Approach"
- Hull, J., & White, A. (2004). "Valuation of a CDO and nth to Default CDS"
- Madan, D. B., & Seneta, E. (1990). "The Variance Gamma (V.G.) Model for Share Market Returns"
