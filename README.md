# Heston model — GPU Monte Carlo (Proj2026)

CUDA implementations for Sorbonne **Calcul haute performance** — Heston section of **Proj2026**.

## Objective

> In this project, we aim to **compare three distinct methods** for simulating an **at-the-money call option** (where “at-the-money” means **K = S₀ = 1**) at maturity **T = 1** under the **Heston model**.

This repository aligns with that goal as follows:

| Method | Program |
|--------|---------|
| Euler discretization | `MC_Euler` |
| Exact variance simulation ([1], [2]) + Step‑3 log-price | `MC_exact` |
| Same variance simulation + **almost-exact** log‑S scheme (Q3) | `MC_almost`, `MC_benchmark_Q3` |

## Google Colab

You may use the notebook below if you want, which clones the repo and runs `make` on a **GPU** runtime.

1. Push this project to GitHub (notebook path must exist on the remote).
2. Replace `OWNER` and `REPO` in the link with your GitHub user/org and repository name; change `main` if your default branch differs.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chezke/Monte-Carlo-simulation-of-Heston-model/blob/main/notebooks/heston_mc_colab.ipynb)

Source notebook: [`notebooks/heston_mc_colab.ipynb`](notebooks/heston_mc_colab.ipynb) — edit `REPO_URL` and `PROJECT_SUBDIR` in the clone cell if your layout differs.

**Limits:** Colab needs a **public** repo for anonymous `git clone`, or you must add credentials. GPU type and CUDA/`nvcc` version depend on Colab’s image; if `make` fails, check `!which nvcc` and `!nvcc --version`.

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

| Binary | Role |
|--------|------|
| `MC_Euler` | Q1–style Euler MC |
| `MC_exact` | Exact variance path + Step-3 log-price (Proj steps 1–3) |
| `MC_almost` | Exact variance + almost-exact **log S** (Q3 scheme) |
| `MC_benchmark_Q3` | Parameter sweep: Euler vs almost-exact timings, **Δt = 1/1000** vs **1/30** |

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

Course code builds on lecture materials; retain author / citation notices in source files where present.
