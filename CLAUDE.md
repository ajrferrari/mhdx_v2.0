# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An HDX-MS (Hydrogen-Deuterium Exchange Mass Spectrometry) isotope analysis pipeline for Waters Synapt HDMS/HDMSE instruments. It processes 4D raw instrument data (RT × DT × m/z × intensity) to extract isotopic envelopes of intact proteins, then optionally identifies them against a protein database.

All compute-heavy processing runs on a SLURM HPC cluster via Snakemake inside a Singularity container. The container embeds the x86-64-only Waters MassLynx SDK — the pipeline **cannot run** on ARM (Apple M-series) without the container.

## Running the Pipeline

### Main isotope pipeline

```bash
# Dry-run to check the DAG
snakemake -s src/Snakefile --configfile src/config.yaml -n

# SLURM cluster submission
snakemake -s src/Snakefile \
  --configfile src/config.yaml \
  -j 1000 --keep-going \
  --use-singularity \
  --singularity-args "--bind /projects/b1107,/scratch/ajf4103" \
  --cluster "sbatch -A p31346 -p short -N 1 -n {resources.cpus} \
             --mem={resources.mem_mb}M -t {resources.runtime} \
             --output results/logs/slurm/slurm-%j.out \
             --error  results/logs/slurm/slurm-%j.err" \
  --max-jobs-per-second 3 --latency-wait 60 --rerun-incomplete
```

Create SLURM log dir before first run: `mkdir -p results/logs/slurm`

### Protein identification pipeline

```bash
snakemake -s src/Snakefile_identify \
  --configfile src/config_identify.yaml \
  -j 500 --keep-going \
  --use-singularity \
  --singularity-args "--bind /projects/b1107,/scratch/ajf4103" \
  --cluster "sbatch -A p31346 -p short -N 1 -n {resources.cpus} \
             --mem={resources.mem_mb}M -t {resources.runtime} \
             --output results/id/logs/slurm-%j.out \
             --error  results/id/logs/slurm-%j.err" \
  --max-jobs-per-second 3 --latency-wait 60 --rerun-incomplete
```

Create log dir: `mkdir -p results/id/logs/slurm`

### Running individual pipeline steps manually (inside the container)

```bash
# Write slice list (checkpoint step)
python src/pipeline.py write_slice_list sample.raw license.key \
  --output_json results/slices/sample/slice_list.json \
  --metrics_csv results/final/sample_slice_metrics.csv

# Process a single batch
python src/pipeline.py process_batch sample.raw license.key \
  --slice_list results/slices/sample/slice_list.json \
  --batch_idx 0 --batch_size 50 \
  --output_dir results/plots/sample \
  --output_csv results/slices/sample/batch_0000.csv

# Aggregate batch CSVs
python src/pipeline.py aggregate \
  --input_pattern 'results/slices/sample/*.csv' \
  --output_csv results/final/sample_isotopes.csv

# Apply filters
python src/pipeline.py apply_filters \
  --input_csv results/final/sample_isotopes_cal.csv \
  --output_csv results/final/sample_isotopes_filtered.csv

# Lock-mass calibration
python src/calibration.py extract sample.raw license.key \
  --output_json results/final/sample_calibration.json \
  --output_pdf results/final/sample_calibration.pdf

python src/calibration.py apply_csv \
  --input_csv results/final/sample_isotopes.csv \
  --cal_json results/final/sample_calibration.json \
  --output_csv results/final/sample_isotopes_cal.csv
```

## Architecture

### Data flow

```
Waters .raw file
    │
    ▼  write_slice_list (checkpoint)
slice_list.json  +  {sample}_slice_metrics.csv
    │
    ▼  process_batch  (ceil(N_slices / batch_size) parallel SLURM jobs)
batch_NNNN.csv   ← one per batch; individual slice errors are caught, batch always writes output
    │
    ▼  aggregate
{sample}_isotopes.csv
    │
    ├──▶  calibrate_lockmass (parallel) ──▶ {sample}_calibration.json
    │
    ▼  apply_calibration
{sample}_isotopes_cal.csv
    │
    ▼  apply_filters
{sample}_isotopes_filtered.csv
    │
    ▼  Snakefile_identify (separate workflow)
{run}/{db}_identifications.csv
```

### Module responsibilities

| File | Role |
|------|------|
| `pipeline.py` | Orchestrates one slice: BPI/TIC check → tensor → NTF → isotope analysis → CSV. Also contains CLI entry points for all Snakemake shell directives. |
| `tensor_analysis.py` | Builds 3D (RT × DT × m/z) tensors; runs NTF (PARAFAC) with automatic rank selection via pairwise correlation; Gaussian quality filters on RT/DT modes. |
| `isotope_analysis.py` | Per-factor isotope envelope detection: peak finding, charge-state assignment, averagine cosine scoring, monoisotopic mass inference. |
| `waters_reader.py` | Wraps Waters MassLynx SDK v5.0.0; provides `WatersRawReader` context manager for accessing RT×DT×m/z data cubes, TIC, and metadata. |
| `calibration.py` | Lock-mass calibration: reads last function of .raw, fits Gaussian peaks to Sodium Formate reference masses, builds per-RT-chunk polynomial curves. |
| `protein_identification.py` | Two-stage database search (50 ppm → polyfit recalibration → 10 ppm) with decoy generation and FDR estimation. |
| `Snakefile` | Main Snakemake DAG: write_slice_list → process_batch → aggregate → calibrate_lockmass → apply_calibration → apply_filters. |
| `Snakefile_identify` | Secondary DAG: generate_decoys → identify (one job per run × database pair). |
| `config.yaml` | All parameters for the main pipeline (slice grid, NTF, thresholds, SLURM resources). |
| `config_identify.yaml` | Parameters for the identification workflow. |

### Key design decisions

**Batched DAG**: Slices are grouped into batches (`batch_size=50` default) to keep the Snakemake DAG manageable. 20,000 slices → 400 DAG nodes instead of 20,000.

**Two-tier filtering**: `pre_filter` (chromatographic TIC, fast) prunes empty RT windows before SLURM submission. `thresholds` (3D BPI/TIC, inside each batch job) prunes silent slices before NTF runs. Units differ by orders of magnitude — do not mix them.

**Deduplication (`is_best`)**: Computed at aggregation time across the full sample. Groups signals by monoisotopic m/z (5 ppm seed-based), keeps highest-BPI per group. A second entry is also marked `is_best=True` if its RT center differs by >30 s (preserves genuine co-eluters at same m/z).

**WatersRawReader**: Must be used as a context manager (`with WatersRawReader(...) as r`). SDK is x86-64 Linux/Windows only — importing on ARM produces a warning and `_SDK_AVAILABLE = False`.

### Output schema

The canonical result DataFrame has these key column groups (defined in `pipeline._RESULT_COLUMNS`):
- Slice coordinates: `rt_lo/hi`, `dt_lo/hi`, `mz_lo/hi`
- Identification: `charge`, `monoisotopic_mz`, `monoisotopic_mass_da`, `cosine_similarity`
- Intensity hierarchy: `bpi/tic` (raw slice) → `factor_bpi/tic` (NTF factor) → `cluster_bpi/tic` (isotope window)
- Quality metrics: `rt_gaussian_r2`, `dt_gaussian_r2`, `peak_rmse`, `fit_quality`
- Deduplication: `is_best`, `k` (0 = monoisotopic row, 1 = isotope peak row)

## Configuration Quick Reference

Edit `src/config.yaml` before launching. Key parameters:

- `samples`: list of absolute paths to `.raw` directories
- `batch_size`: slices per SLURM job (default 50; max ~80 for 4h short partition)
- `thresholds.bpi_min` / `thresholds.tic_min`: 3D signal quality cuts (start at 0, tune from results)
- `ntf.rank_init` / `ntf.rank_max`: NTF component count bounds
- `plots.save_factors`: false by default; enable only for QC (large storage cost)
- `calibration`: lock-mass compound and polynomial degree
