/*
 * Shared Monte Carlo launch settings for Heston exercises (Euler, exact, etc.).
 * Override at compile time, e.g.:
 *   nvcc -DHESTON_MC_N_PATHS=524288 -DHESTON_MC_CURAND_SEED=7ULL ...
 */

#ifndef HESTON_MC_COMMON_CUH
#define HESTON_MC_COMMON_CUH

#ifndef HESTON_RHO
#define HESTON_RHO (-0.7f)
#endif

#ifndef HESTON_MC_N_PATHS
#define HESTON_MC_N_PATHS (262144)
#endif

#ifndef HESTON_MC_THREADS_PER_BLOCK
#define HESTON_MC_THREADS_PER_BLOCK (512)
#endif

/* Same seed for all drivers so path counts match when comparing estimators. */
#ifndef HESTON_MC_CURAND_SEED
#define HESTON_MC_CURAND_SEED (12345ULL)
#endif

/* MC_benchmark_Q3.cu: optional overrides
 *   -DHESTON_Q3_N_PATHS=65536
 *   -DHESTON_Q3_GRID_K=10 -DHESTON_Q3_GRID_T=10 -DHESTON_Q3_GRID_S=10
 */

#endif /* HESTON_MC_COMMON_CUH */
