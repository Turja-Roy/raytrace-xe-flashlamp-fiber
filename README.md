# VUV Flashlamp-to-Fiber Coupling Optimization

A comprehensive ray tracing and optimization framework for designing two-lens plano-convex optical systems that efficiently couple vacuum ultraviolet (VUV) light from xenon flashlamp sources into optical fibers.

## Project Overview

This project addresses the challenge of efficiently collecting 200nm light from a xenon flashlamp arc source and coupling it into a small-core optical fiber. The system uses:

- **Source**: 60W Xenon flash lamp with 3mm diameter arc and 66° divergence angle
- **Target**: 1mm core optical fiber with NA=0.22 and acceptance angle of 24.8°
- **Approach**: Two-lens plano-convex system optimization via deterministic ray tracing
- **Physics**: Full geometric optics with atmospheric O₂ absorption modeling

### Key Results

- **Coupling Efficiency**: 0.19-0.24 in air, 0.24-0.29 in argon for optimized configurations
- **System Length**: 33-95mm depending on lens selection
- **Atmospheric Impact**: ~8% coupling improvement in argon vs. air (200nm O₂ absorption elimination)
- **Medium Dependence**: Argon shows measurably better performance than air due to lack of UV absorption

## Features

### Six Optimization Algorithms

| Method | Speed | Convergence | Best For |
|--------|-------|-------------|----------|
| **Powell** ⚡ | 1-2s | Good | **Quick optimization, everyday use** |
| **Nelder-Mead** ⚡ | 1-2s | Good | **Fast local search** |
| Grid Search | 2-3s | Systematic | Baseline comparison, small grids |
| Differential Evolution | 10-17s | Excellent | Thorough global search when time permits |
| Bayesian | 20-22s | Excellent | Sample-efficient exploration |
| Dual Annealing | 40-51s | Excellent | Escaping local minima |

*Timings are per lens pair with 1000 rays on typical hardware*

### Multi-Objective Optimization

Balance coupling efficiency against system compactness using the alpha parameter:

```
objective = α × (1 - coupling) + (1 - α) × (normalized_length)
```

- **α = 0.9**: Maximize coupling (90% weight) - laboratory setups
- **α = 0.7**: Balanced (default) - general purpose
- **α = 0.5**: Equal priority - compact designs

### Additional Capabilities

- **Wavelength Analysis**: Study coupling efficiency across 180-300nm range
- **Medium Selection**: Air, argon, or helium propagation
- **Resume Support**: Automatic checkpoint/resume for interrupted batch runs
- **Rich Visualization**: Ray trace diagrams, spot diagrams, wavelength plots
- **Batch Processing**: Automatic splitting and parallel processing of large lens catalogs

## Installation

### Requirements

- Python 3.7+
- Required packages:
  ```bash
  pip install numpy scipy pandas matplotlib tqdm regex
  ```
- Optional (for Bayesian optimization):
  ```bash
  pip install scikit-optimize
  ```

### Quick Start

```bash
git clone <repository>
cd raytrace-xe-flashlamp-fiber
python raytrace.py particular LA4001 LA4647 --opt powell
```

## Usage Guide

### Basic Commands

#### Optimize a Specific Lens Pair (Recommended Starting Point)

```bash
# Fast optimization with Powell's method (recommended)
python raytrace.py particular LA4001 LA4647 --opt powell

# Thorough optimization with differential evolution (slower but more thorough)
python raytrace.py particular LA4001 LA4647 --opt differential_evolution

# Try in argon (no UV absorption)
python raytrace.py particular LA4001 LA4647 --opt powell --medium argon
```

#### Optimize All Combinations

```bash
# Strategic scan with pre-selected L1/L2 candidates (924 combinations, recommended)
python raytrace.py select --opt powell

# Exhaustive scan of all possible combinations (24,336 combinations)
python raytrace.py combine --opt powell

# Prioritize coupling efficiency (90% weight)
python raytrace.py select --opt powell --alpha 0.9

# Prioritize compact systems (50/50 weight)
python raytrace.py select --opt powell --alpha 0.5
```

**Select vs Combine modes:**
- **select**: Tests 21 L1 candidates × 44 L2 candidates = 924 strategically chosen lens pairs
- **combine**: Tests all 156 lenses × 156 lenses = 24,336 possible combinations (26× more)
- **Recommendation**: Use `select` for efficient exploration; use `combine` only when exhaustive search is needed

#### Compare All Methods

```bash
# Test all 6 optimization methods on a single lens pair
python raytrace.py compare LA4001 LA4647
```

This runs all available methods and reports:
- Coupling efficiency achieved
- Total system length
- Computation time
- Best method recommendation

#### Resume Interrupted Batch Runs

```bash
# Resume from checkpoint
python raytrace.py combine --opt powell continue 2025-10-14
```

#### Analyze High-Coupling Results

```bash
# Re-optimize previously found good configurations with all methods
python raytrace.py analyze \
  --results-file results/2025-10-17/results_combine_*.csv \
  --coupling-threshold 0.20 \
  --opt powell
```

#### Wavelength Dependence Analysis

```bash
# Analyze coupling vs wavelength for specific lens pairs
python raytrace.py wavelength-analyze \
  --results-file results/2025-10-17/36-681+LA4647.csv \
  --wl-start 180 --wl-end 300 --wl-step 10

# Custom range with more rays for precision
python raytrace.py wavelength-analyze \
  --results-file results/2025-10-17/36-681+LA4647.csv \
  --wl-start 190 --wl-end 210 --wl-step 2 \
  --n-rays 5000

# Generate plots from wavelength analysis
python raytrace.py wavelength-analyze-plot \
  --results-dir results/wavelength_analyze_2025-10-18

# With polynomial curve fitting
python raytrace.py wavelength-analyze-plot \
  --results-dir results/wavelength_analyze_2025-10-18 \
  --fit polynomial

# Aggregated plots with error bars
python raytrace.py wavelength-analyze-plot \
  --results-dir results/wavelength_analyze_2025-10-18 \
  --aggregate
```

### Command Reference

```
Commands:
  particular <lens1> <lens2>    Optimize specific lens pair
  compare <lens1> <lens2>       Compare all optimization methods
  select                        Optimize L1×L2 candidate combinations (924 pairs, recommended)
  combine                       Optimize all lens combinations (24,336 pairs, exhaustive)
  analyze                       Re-analyze high-coupling results
  wavelength-analyze            Study wavelength dependence
  wavelength-analyze-plot       Create plots from wavelength data

Options:
  --opt <method>                Optimization method (default: powell)
                                Options: powell, nelder_mead, grid_search,
                                         differential_evolution, bayesian,
                                         dual_annealing
  --alpha <0-1>                 Coupling vs. length trade-off (default: 0.7)
  --medium <type>               Propagation medium (default: air)
                                Options: air, argon, helium
  --n-rays <count>              Number of rays per trace (default: 1000)
  --wl-start <nm>               Wavelength analysis start (default: 180)
  --wl-end <nm>                 Wavelength analysis end (default: 300)
  --wl-step <nm>                Wavelength analysis step (default: 10)
  --coupling-threshold <value>  Minimum coupling for analyze mode
  --results-file <path>         Input CSV file for analyze/wavelength modes
  --results-dir <path>          Results directory for plotting
  --fit <type>                  Curve fit type: polynomial, spline
  --aggregate                   Generate aggregated plots with error bars
  continue                      Resume incomplete batch run
  <YYYY-MM-DD>                  Specify run date for continue mode
```

## File Structure

```
raytrace-xe-flashlamp-fiber/
├── raytrace.py                    # Main entry point
│
├── scripts/
│   ├── optimization/              # Optimization algorithms
│   │   ├── __init__.py
│   │   ├── powell.py              # Powell's method (fast)
│   │   ├── nelder_mead.py         # Nelder-Mead simplex (fast)
│   │   ├── grid_search.py         # Systematic grid search
│   │   ├── differential_evolution.py  # Global optimizer (thorough)
│   │   ├── bayesian.py            # Bayesian optimization
│   │   ├── dual_annealing.py      # Simulated annealing
│   │   └── optimization_runner.py # Unified interface
│   │
│   ├── analysis.py                # Analysis and wavelength studies
│   ├── calcs.py                   # Physics calculations (refraction, absorption)
│   ├── cli.py                     # Command-line interface
│   ├── consts.py                  # Physical constants and system parameters
│   ├── data_io.py                 # File I/O and result management
│   ├── hitran_data.py             # HITRAN absorption data handling
│   ├── PlanoConvex.py             # Plano-convex lens class
│   ├── raytrace_helpers.py        # Core ray tracing functions
│   ├── runner.py                  # Batch processing and continuation
│   └── visualizers.py             # Plotting and visualization
│
├── data/                          # Lens catalogs
│   ├── Combined_Lenses.csv        # Full lens catalog (156 lenses)
│   ├── l1_candidates.csv          # Candidate L1 lenses
│   ├── l2_candidates.csv          # Candidate L2 lenses
│   └── *.csv, *.xlsx              # Vendor lens data
│
├── results/                       # Output directory
│   └── <run_id>/                  # Results organized by run ID
│       ├── batch_*.csv            # Intermediate batch results
│       ├── results_*.csv          # Final combined results
│       └── temp_*.json            # Checkpoint files
│
├── plots/                         # Generated plots
│   └── <run_id>/
│       ├── C-<coupling>_L1-*_L2-*.png      # Ray trace diagrams
│       ├── spot_*.png                       # Spot diagrams
│       └── wavelength_analyze_*/            # Wavelength analysis plots
│
├── logs/                          # Execution logs
│   └── *.log                      # Detailed execution logs
│
└── doc/                           # Documentation
    ├── technical_report.pdf       # Full technical report
    └── *.tex, *.bib               # LaTeX source files
```

## Algorithm Details

### Powell's Method (Recommended Default)

**Type**: Direction-set local optimizer  
**Speed**: 1-2 seconds per lens pair  
**Convergence**: Good, finds local optima reliably  
**Pros**: Very fast, no gradient needed, efficient  
**Cons**: May miss global optimum if started far away  
**Best for**: Everyday optimization, quick scans, general use

### Nelder-Mead

**Type**: Simplex-based local optimizer  
**Speed**: 1-2 seconds per lens pair  
**Convergence**: Good, geometric search  
**Pros**: Very fast, robust, no gradient needed  
**Cons**: Can be sensitive to initial conditions  
**Best for**: Quick local refinement, alternative to Powell

### Grid Search

**Type**: Systematic parameter sweep  
**Speed**: 2-3 seconds per lens pair (with default 7×7 grid)  
**Convergence**: Systematic, guaranteed to check all grid points  
**Pros**: Reproducible, visualizes parameter space  
**Cons**: Fixed resolution, computationally intensive for fine grids  
**Best for**: Baseline comparisons, parameter space visualization

### Differential Evolution

**Type**: Population-based global optimizer  
**Speed**: 10-17 seconds per lens pair  
**Convergence**: Excellent, finds global optima consistently  
**Pros**: Robust, explores full parameter space, no gradient needed  
**Cons**: Slower than local methods  
**Best for**: Final optimization when thoroughness matters, difficult problems

### Bayesian Optimization

**Type**: Sequential model-based optimization  
**Speed**: 20-22 seconds per lens pair  
**Convergence**: Excellent, sample-efficient  
**Pros**: Models uncertainty, very sample-efficient, good for expensive evaluations  
**Cons**: Requires scikit-optimize, slower than Powell  
**Requires**: `pip install scikit-optimize`  
**Best for**: When function evaluations are expensive, uncertainty quantification

### Dual Annealing

**Type**: Hybrid simulated annealing with local search  
**Speed**: 40-51 seconds per lens pair  
**Convergence**: Excellent, escapes local minima  
**Pros**: Very thorough exploration, combines global and local search  
**Cons**: Slowest method  
**Best for**: Complex landscapes with many local optima, research purposes

## Output Files

### Results CSVs

All optimization results are saved as CSV files with columns:

- `lens1`, `lens2`: Lens identifiers
- `f1_mm`, `f2_mm`: Focal lengths (mm)
- `z_l1`, `z_l2`, `z_fiber`: Positions along optical axis (mm)
- `total_len_mm`: Total system length (mm)
- `coupling`: Coupling efficiency (0-1)

Example: `results/2025-10-18_combine_powell_air/results_combine_2025-10-18.csv`

### Plot Files

#### Ray Trace Diagrams
Shows full ray paths through the optical system with lens positions and fiber location.

Filename format: `C-<coupling>_L1-<lens1>_L2-<lens2>.png`

#### Spot Diagrams
Shows ray impact positions at the fiber face, indicating coupling quality.

Filename format: `spot_C-<coupling>_L1-<lens1>_L2-<lens2>.png`

#### Wavelength Analysis Plots
- Per-lens plots: Coupling vs. wavelength for each lens combination
- Per-method plots: Comparing all lens pairs for each optimization method
- Aggregated plots: Mean curves with error bars across multiple runs

### Log Files

Detailed execution logs including:
- Timestamp for each optimization
- Method being used
- Coupling efficiency and system length
- Computation time
- Error messages and warnings

Example: `logs/2025-10-18_combine_powell_air.log`

## Physics and Implementation

### Ray Tracing

Deterministic ray tracing with quasi-random sampling:
- Rays generated from finite 3mm arc source (radial positions randomly sampled)
- Angular positions evenly distributed around circle
- Ray angles deterministically calculated based on radial position within 33° cone
- Full refraction at each lens surface (Snell's law)
- Acceptance testing at fiber based on NA and position

**Note**: Despite the historical filename `raytrace_monte-carlo.py`, this uses stratified sampling rather than pure Monte Carlo. Only radial positions are randomized; angular positions and ray directions are deterministic.

### Atmospheric Absorption

O₂ absorption at 200nm modeled using Minschwaner parameterization:
- **Air (21% O₂)**: ~8% coupling loss vs. argon (measured)
- **Argon/Helium**: Negligible absorption at 200nm, ~8% better coupling than air
- Temperature, pressure, and humidity dependent

### Multi-Objective Optimization

The objective function balances two competing goals:

```python
objective = alpha * (1 - coupling) + (1 - alpha) * (length / max_length)
```

Where:
- `coupling`: Fraction of rays accepted by fiber (0-1)
- `length`: Total optical system length (mm)
- `alpha`: User-controlled weight (0-1)

Higher alpha prioritizes coupling; lower alpha prioritizes compactness.

## Performance Recommendations

### For Everyday Use
**Use Powell's method** - Fast and reliable:
```bash
python raytrace.py combine --opt powell
```

### For Best Possible Results
**Use Differential Evolution** - Slower but more thorough:
```bash
python raytrace.py combine --opt differential_evolution
```

### For Method Comparison
**Use compare mode** - Test all methods on a sample:
```bash
python raytrace.py compare LA4001 LA4647
```

### For Large Catalogs
**Start with Powell, verify with DE**:
```bash
# 1. Fast scan with Powell
python raytrace.py combine --opt powell

# 2. Re-optimize top candidates with differential evolution
python raytrace.py analyze \
  --results-file results/2025-10-18/results_*.csv \
  --coupling-threshold 0.20 \
  --opt differential_evolution
```

## Example Workflows

### Workflow 1: Find Best Lens Pair for Maximum Coupling

```bash
# Step 1: Quick scan of all combinations
python raytrace.py combine --opt powell --alpha 0.9

# Step 2: Check results
head -20 results/2025-10-18_combine_powell_air/results_combine_2025-10-18.csv

# Step 3: Re-optimize top candidates more thoroughly
python raytrace.py analyze \
  --results-file results/2025-10-18_combine_powell_air/results_combine_2025-10-18.csv \
  --coupling-threshold 0.22 \
  --opt differential_evolution \
  --alpha 0.9

# Step 4: Review final results
cat results/analyze_2025-10-18_coupling_0_22_air/results_analyze_combined_*.csv
```

### Workflow 2: Design Compact System

```bash
# Optimize with equal weight to coupling and compactness
python raytrace.py combine --opt powell --alpha 0.5

# Find shortest systems with acceptable coupling
python raytrace.py analyze \
  --results-file results/2025-10-18_combine_powell_air/results_combine_2025-10-18.csv \
  --coupling-threshold 0.18 \
  --opt powell \
  --alpha 0.3
```

### Workflow 3: Wavelength Characterization

```bash
# Step 1: Find best configuration at 200nm
python raytrace.py particular 36-681 LA4647 --opt differential_evolution

# Step 2: Analyze wavelength dependence
python raytrace.py wavelength-analyze \
  --results-file results/particular_2025-10-18_differential_evolution_air/36-681+LA4647.csv \
  --wl-start 180 --wl-end 250 --wl-step 5 \
  --n-rays 2000

# Step 3: Generate plots
python raytrace.py wavelength-analyze-plot \
  --results-dir results/wavelength_analyze_2025-10-18 \
  --fit polynomial
```

### Workflow 4: Compare Air vs. Argon

```bash
# Run in air
python raytrace.py particular 36-681 LA4647 --opt powell --medium air

# Run in argon
python raytrace.py particular 36-681 LA4647 --opt powell --medium argon

# Compare coupling efficiencies in logs
grep "Coupling=" logs/particular_2025-10-18_powell_air.log
grep "Coupling=" logs/particular_2025-10-18_powell_argon.log
```

## Troubleshooting

### Slow Performance

**Problem**: Optimization takes too long  
**Solutions**:
- Use Powell or Nelder-Mead instead of Differential Evolution
- Reduce number of rays: `--n-rays 500`
- Process specific lens pairs instead of full catalog

### Poor Coupling Results

**Problem**: Low coupling efficiency  
**Solutions**:
- Try different optimization method: `--opt differential_evolution`
- Increase alpha to prioritize coupling: `--alpha 0.9`
- Verify lens data in `data/Combined_Lenses.csv`
- Check if lenses are physically compatible

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'skopt'`  
**Solution**: Install scikit-optimize or use different method:
```bash
pip install scikit-optimize
# OR
python raytrace.py combine --opt powell
```

### Interrupted Batch Runs

**Problem**: Run stopped midway through catalog  
**Solution**: Resume from checkpoint:
```bash
python raytrace.py combine --opt powell continue 2025-10-18
```

### Memory Issues

**Problem**: Out of memory with large catalogs  
**Solutions**:
- Batch processing is automatic for >100 combinations
- Reduce rays per trace: `--n-rays 500`
- Process smaller subsets using `particular` mode

## Technical Background

### Research Context

This project implements the optimization framework described in:

> **"Optimization of Two-Lens Coupling Systems for VUV Flashlamp to Fiber Applications Using Ray Tracing and Multi-Algorithm Comparison"**  
> Turja Roy, Department of Physics, University of Texas at Arlington

Key findings:
- Coupling efficiencies of 0.19-0.24 in air, 0.24-0.29 in argon achievable with optimized configurations
- Argon provides ~8% coupling improvement over air due to eliminated O₂ absorption
- Powell's method is fastest (1-2s) while differential evolution is most thorough (10-17s)
- Multi-objective optimization successfully balances efficiency and compactness

### Physical Constants

Defined in `scripts/consts.py`:
- Wavelength: 200nm (VUV range)
- Fiber: 1mm core, NA=0.22
- Source: 3mm arc diameter, 33° maximum angle
- Medium properties: Temperature, pressure, humidity dependent
- Lens material: Fused silica (refractive index from Sellmeier equation)

### Optimization Implementation

Each optimizer implements:
```python
def optimize(lenses, lens1_name, lens2_name, n_rays=1000, alpha=0.7, medium='air'):
    # Returns: dict with keys: lens1, lens2, z_l1, z_l2, z_fiber, 
    #          coupling, total_len_mm, f1_mm, f2_mm, origins, dirs, accepted
```

Parameter bounds:
- `z_l1`: 9.7 - 12.0 mm (lens 1 position)
- `z_l2`: 9.7 - 50.0 mm (lens 2 position, constrained > z_l1 + lens thickness)
- `z_fiber`: Automatically optimized based on focal length and magnification

## Citation

If you use this code in your research, please cite:

```
Turja Roy, "Optimization of Two-Lens Coupling Systems for VUV Flashlamp 
to Fiber Applications Using Ray Tracing and Multi-Algorithm Comparison",
University of Texas at Arlington (2025)
```

Technical report: `doc/technical_report.pdf`

## Author

**Turja Roy**  
Department of Physics  
University of Texas at Arlington

## Contributing

For questions, issues, or contributions, please contact the author or open an issue in the repository.
