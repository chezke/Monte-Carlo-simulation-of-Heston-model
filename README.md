# Heston model — GPU Monte Carlo (Proj2026)

CUDA implementations for Sorbonne **Calcul haute performance** — Heston section of **Proj2026**.

## Objective

> In this project, we aim to **compare three distinct methods** for simulating an **at-the-money call option** (where “at-the-money” means **$K = S_0 = 1$**) at maturity **T = 1** under the **Heston model**.

The quantity of interest is: $E[(S_1 - 1)_+]$.

This repository aligns with that goal as follows:

| Method | Program |
|--------|---------|
| Euler discretization | `MC_Euler` |
| Exact simulation of variance (Broadie–Kaya) | `MC_exact` |
| Same variance simulation + almost-exact log-price scheme | `MC_almost`, `MC_benchmark_Q3` |

## Optional: run on GPU via Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chezke/Monte-Carlo-simulation-of-Heston-model/blob/main/notebooks/heston_mc_colab.ipynb)

Source notebook: [`notebooks/heston_mc_colab.ipynb`](notebooks/heston_mc_colab.ipynb).

## Layout

| Path | Content |
|------|---------|
| [`include/`](include/) | Shared device headers (`*.cuh`): RNG helpers, CIR exact step, Gamma [1], MC launch macros |
| [`src/`](src/) | Programs (`*.cu`) |
| [`bin/`](bin/) | Compiled binaries (generated; gitignored) |

## Requirements

- NVIDIA **CUDA Toolkit** (`nvcc`, cuRAND)

## Build

From this directory:

```bash
make
```

Executables are written to `bin/`:

| Binary | Description |
|--------|-------------|
| `MC_Euler` | Euler discretization Monte Carlo (Q1) |
| `MC_exact` | Exact simulation of variance (Broadie–Kaya), including steps 1–3: variance sampling, integral computation, and log-price reconstruction (Q2) |
| `MC_almost` | Almost-exact log-price scheme based on exact variance simulation (Q3) |
| `MC_benchmark_Q3` | Benchmark comparing Euler and almost-exact methods, including execution time and step size comparison (**Δt = 1/1000** vs **1/30**) |

### Manual `nvcc` (equivalent)

```bash
mkdir -p bin
nvcc -O3 -std=c++14 -Iinclude src/MC_Euler.cu       -o bin/MC_Euler
nvcc -O3 -std=c++14 -Iinclude src/MC_exact.cu       -o bin/MC_exact
nvcc -O3 -std=c++14 -Iinclude src/MC_almost.cu      -o bin/MC_almost
nvcc -O3 -std=c++14 -Iinclude src/MC_benchmark_Q3.cu -o bin/MC_benchmark_Q3
```

### Common compile-time defines

Defined in [`include/heston_mc_common.cuh`](include/heston_mc_common.cuh), overridable on the command line, e.g.:

```bash
make NVFLAGS="-O3 -std=c++14 -Iinclude -DHESTON_MC_N_PATHS=65536 -DHESTON_RHO=-0.5f"
```

## Run

```bash
./bin/MC_Euler
./bin/MC_exact
./bin/MC_almost
./bin/MC_benchmark_Q3 > results_q3.csv
```

## References (course)

- **[1]** G. Marsaglia & W. W. Tsang, *A simple method for generating gamma variables*, ACM TOMS 26(3), 2000.  
- **[2]** M. Broadie & Ö. Kaya, *Exact simulation of stochastic volatility…*, *Operations Research* 54(2), 2006.

## License / attribution

Based on course materials (instructor: *Lokman A. Abbas-Turki*). **Keep author / course notices** already present in borrowed or adapted source files.
