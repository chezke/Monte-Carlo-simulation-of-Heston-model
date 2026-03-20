/*
 * Proj2026 — Question 3: compare GPU times Euler vs almost-exact (exact variance),
 * on a (kappa, theta, sigma) grid with Feller 2*kappa*theta > sigma^2.
 * Also times almost-exact with dt=1/1000 vs dt=1/30.
 *
 * CSV columns:
 *   id,kappa,theta,sigma,feller_lhs_minus_rhs,ms_euler,ms_almost_dt1000,ms_almost_dt1_30,
 *   mean_euler,mean_almost_1000,mean_almost_30
 *
 * Build: see repository Makefile / README.md (`make`, outputs `bin/MC_benchmark_Q3`).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include "heston_mc_common.cuh"
#include "heston_cuda_utils.cuh"
#include "heston_cir_exact.cuh"

#ifndef USE_ABS_VARIANCE_TRUNC
#define USE_ABS_VARIANCE_TRUNC 0
#endif

#ifndef HESTON_Q3_N_PATHS
#define HESTON_Q3_N_PATHS HESTON_MC_N_PATHS
#endif

#ifndef HESTON_Q3_GRID_K
#define HESTON_Q3_GRID_K (8)
#endif
#ifndef HESTON_Q3_GRID_T
#define HESTON_Q3_GRID_T (8)
#endif
#ifndef HESTON_Q3_GRID_S
#define HESTON_Q3_GRID_S (8)
#endif

#define Q3_KAPPA_MIN 0.1f
#define Q3_KAPPA_MAX 10.0f
#define Q3_THETA_MIN 0.01f
#define Q3_THETA_MAX 0.5f
#define Q3_SIGMA_MIN 0.1f
#define Q3_SIGMA_MAX 1.0f

#define DT_EULER (1.0f / 1000.0f)
#define DT_ALMOST_FINE (1.0f / 1000.0f)
#define DT_ALMOST_COARSE (1.0f / 30.0f)

__global__ void init_curand_state_k(curandState* state, unsigned long long seed) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__forceinline__ __device__ float g_trunc(float x) {
#if USE_ABS_VARIANCE_TRUNC
	return fabsf(x);
#else
	return fmaxf(x, 0.0f);
#endif
}

__device__ float heston_euler_path_payoff(
	curandState* local,
	float S0, float v0, float r, float K,
	float dt, int n_steps,
	float kappa, float theta, float sigma, float rho) {

	float S = S0;
	float v = v0;
	float sqrt_dt = sqrtf(dt);
	float rho_sq = rho * rho;
	float sqrt_1m_r2 = sqrtf(fmaxf(0.0f, 1.0f - rho_sq));

	for (int step = 0; step < n_steps; step++) {
		float2 G = curand_normal2(local);
		float G1 = G.x;
		float G2 = G.y;
		float Zhat = rho * G1 + sqrt_1m_r2 * G2;
		float sqrt_v = sqrtf(fmaxf(v, 0.0f));
		S = S + r * S * dt + sqrt_v * S * sqrt_dt * Zhat;
		float v_pred = v + kappa * (theta - v) * dt + sigma * sqrt_v * sqrt_dt * G1;
		v = g_trunc(v_pred);
	}
	return fmaxf(0.0f, S - K);
}

__global__ void heston_euler_mc_k(
	float S0, float v0, float r, float K,
	float dt, int n_steps,
	float kappa, float theta, float sigma, float rho,
	curandState* state,
	float* sum, int n_paths) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	float payoff = 0.f;
	if (idx < n_paths) {
		curandState localState = state[idx];
		payoff = heston_euler_path_payoff(
			&localState, S0, v0, r, K, dt, n_steps,
			kappa, theta, sigma, rho);
		state[idx] = localState;
	}
	extern __shared__ float sh[];
	float* R1 = sh;
	float* R2 = R1 + blockDim.x;
	R1[threadIdx.x] = payoff / (float)n_paths;
	R2[threadIdx.x] = R1[threadIdx.x] * R1[threadIdx.x] * (float)n_paths;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			R1[threadIdx.x] += R1[threadIdx.x + stride];
			R2[threadIdx.x] += R2[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicAdd(sum, R1[0]);
		atomicAdd(sum + 1, R2[0]);
	}
}

__device__ float heston_almost_path_payoff(
	curandState* local,
	float S0, float v0, float r, float K,
	float dt, int n_steps,
	float kappa, float theta, float sigma, float rho) {

	float logS = logf(S0);
	float v = v0;
	const float rho_os = rho / sigma;
	const float k0 = -rho_os * kappa * theta * dt;
	const float k1 = (rho_os * kappa - 0.5f) * dt - rho_os;
	const float k2 = rho_os;
	const float one_m_r2 = fmaxf(0.f, 1.f - rho * rho);

	for (int step = 0; step < n_steps; step++) {
		float v_old = v;
		v = cir_exact_variance_step(local, v_old, kappa, theta, sigma, dt);
		float G2 = curand_normal(local);
		logS += k0 + k1 * v_old + k2 * v + sqrtf(one_m_r2 * v_old * dt) * G2;
	}
	float drift_log = r * dt * (float)n_steps;
	float S1 = expf(logS + drift_log);
	return fmaxf(0.f, S1 - K);
}

__global__ void heston_almost_mc_k(
	float S0, float v0, float r, float K,
	float dt, int n_steps,
	float kappa, float theta, float sigma, float rho,
	curandState* state,
	float* sum, int n_paths) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	float payoff = 0.f;
	if (idx < n_paths) {
		curandState localState = state[idx];
		payoff = heston_almost_path_payoff(
			&localState, S0, v0, r, K, dt, n_steps,
			kappa, theta, sigma, rho);
		state[idx] = localState;
	}
	extern __shared__ float sh[];
	float* R1 = sh;
	float* R2 = R1 + blockDim.x;
	R1[threadIdx.x] = payoff / (float)n_paths;
	R2[threadIdx.x] = R1[threadIdx.x] * R1[threadIdx.x] * (float)n_paths;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			R1[threadIdx.x] += R1[threadIdx.x + stride];
			R2[threadIdx.x] += R2[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicAdd(sum, R1[0]);
		atomicAdd(sum + 1, R2[0]);
	}
}

static float lerp(float a, float b, float t) {
	return a + (b - a) * t;
}

static void run_timed_euler(
	float S0, float v0, float r, float K,
	float dt, int n_steps,
	float kappa, float theta, float sigma, float rho,
	curandState* states, int nb, int ntpb, int n_paths,
	float* sum,
	unsigned long long seed_tag,
	float* out_mean,
	float* out_ms) {

	TEST_CUDA(cudaMemset(sum, 0, 2 * sizeof(float)));
	init_curand_state_k<<<nb, ntpb>>>(states, HESTON_MC_CURAND_SEED + seed_tag);
	TEST_CUDA(cudaGetLastError());
	TEST_CUDA(cudaDeviceSynchronize());

	cudaEvent_t ev0, ev1;
	TEST_CUDA(cudaEventCreate(&ev0));
	TEST_CUDA(cudaEventCreate(&ev1));
	size_t shmem = 2 * (size_t)ntpb * sizeof(float);

	TEST_CUDA(cudaEventRecord(ev0));
	heston_euler_mc_k<<<nb, ntpb, shmem>>>(
		S0, v0, r, K, dt, n_steps, kappa, theta, sigma, rho, states, sum, n_paths);
	TEST_CUDA(cudaGetLastError());
	TEST_CUDA(cudaDeviceSynchronize());
	TEST_CUDA(cudaEventRecord(ev1));
	TEST_CUDA(cudaEventSynchronize(ev1));
	float ms = 0.f;
	TEST_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
	*out_ms = ms;
	*out_mean = sum[0];
	TEST_CUDA(cudaEventDestroy(ev0));
	TEST_CUDA(cudaEventDestroy(ev1));
}

static void run_timed_almost(
	float S0, float v0, float r, float K,
	float dt, int n_steps,
	float kappa, float theta, float sigma, float rho,
	curandState* states, int nb, int ntpb, int n_paths,
	float* sum,
	unsigned long long seed_tag,
	float* out_mean,
	float* out_ms) {

	TEST_CUDA(cudaMemset(sum, 0, 2 * sizeof(float)));
	init_curand_state_k<<<nb, ntpb>>>(states, HESTON_MC_CURAND_SEED + seed_tag + 5000000ULL);
	TEST_CUDA(cudaGetLastError());
	TEST_CUDA(cudaDeviceSynchronize());

	cudaEvent_t ev0, ev1;
	TEST_CUDA(cudaEventCreate(&ev0));
	TEST_CUDA(cudaEventCreate(&ev1));
	size_t shmem = 2 * (size_t)ntpb * sizeof(float);

	TEST_CUDA(cudaEventRecord(ev0));
	heston_almost_mc_k<<<nb, ntpb, shmem>>>(
		S0, v0, r, K, dt, n_steps, kappa, theta, sigma, rho, states, sum, n_paths);
	TEST_CUDA(cudaGetLastError());
	TEST_CUDA(cudaDeviceSynchronize());
	TEST_CUDA(cudaEventRecord(ev1));
	TEST_CUDA(cudaEventSynchronize(ev1));
	float ms = 0.f;
	TEST_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
	*out_ms = ms;
	*out_mean = sum[0];
	TEST_CUDA(cudaEventDestroy(ev0));
	TEST_CUDA(cudaEventDestroy(ev1));
}

int main(void) {
	const float S0 = 1.f;
	const float v0 = 0.1f;
	const float r = 0.f;
	const float K = 1.f;
	const float T = 1.f;
	const float rho = HESTON_RHO;

	const int n_paths = HESTON_Q3_N_PATHS;
	const int ntpb = HESTON_MC_THREADS_PER_BLOCK;
	const int nb = (n_paths + ntpb - 1) / ntpb;
	const int grid_threads = nb * ntpb;

	float* sum = nullptr;
	TEST_CUDA(cudaMallocManaged(&sum, 2 * sizeof(float)));
	curandState* states = nullptr;
	TEST_CUDA(cudaMalloc(&states, (size_t)grid_threads * sizeof(curandState)));

	const int n_euler = (int)lroundf(T / DT_EULER);
	const int n_af = (int)lroundf(T / DT_ALMOST_FINE);
	const int n_ac = (int)lroundf(T / DT_ALMOST_COARSE);

	const int NK = HESTON_Q3_GRID_K;
	const int NT = HESTON_Q3_GRID_T;
	const int NS = HESTON_Q3_GRID_S;

	printf("# Proj2026 Q3 benchmark  Feller: 2*kappa*theta > sigma^2\n");
	printf("# Euler dt=%g n_steps=%d | almost fine dt=%g n=%d | almost coarse dt=%g n=%d\n",
		(double)DT_EULER, n_euler, (double)DT_ALMOST_FINE, n_af, (double)DT_ALMOST_COARSE, n_ac);
	printf("# paths=%d rho=%g\n", n_paths, (double)rho);
	printf("id,kappa,theta,sigma,feller_gap,ms_euler,ms_almost_dt1000,ms_almost_dt1_30,mean_euler,mean_almost_1000,mean_almost_30\n");

	int id = 0;
	for (int ik = 0; ik < NK; ik++) {
		float tk = (NK <= 1) ? 0.f : (float)ik / (float)(NK - 1);
		float kappa = lerp(Q3_KAPPA_MIN, Q3_KAPPA_MAX, tk);
		for (int it = 0; it < NT; it++) {
			float tt = (NT <= 1) ? 0.f : (float)it / (float)(NT - 1);
			float theta = lerp(Q3_THETA_MIN, Q3_THETA_MAX, tt);
			for (int is = 0; is < NS; is++) {
				float ts = (NS <= 1) ? 0.f : (float)is / (float)(NS - 1);
				float sigma = lerp(Q3_SIGMA_MIN, Q3_SIGMA_MAX, ts);

				float feller_lhs = 2.f * kappa * theta;
				float feller_rhs = sigma * sigma;
				if (!(feller_lhs > feller_rhs))
					continue;

				unsigned long long tag = (unsigned long long)(id + 1) * 100003ULL;

				float ms_e = 0.f, ms_a1 = 0.f, ms_a30 = 0.f;
				float mean_e = 0.f, mean_a1 = 0.f, mean_a30 = 0.f;

				run_timed_euler(S0, v0, r, K, DT_EULER, n_euler,
					kappa, theta, sigma, rho, states, nb, ntpb, n_paths,
					sum, tag, &mean_e, &ms_e);
				run_timed_almost(S0, v0, r, K, DT_ALMOST_FINE, n_af,
					kappa, theta, sigma, rho, states, nb, ntpb, n_paths, sum,
					tag, &mean_a1, &ms_a1);
				run_timed_almost(S0, v0, r, K, DT_ALMOST_COARSE, n_ac,
					kappa, theta, sigma, rho, states, nb, ntpb, n_paths, sum,
					tag + 7777777ULL, &mean_a30, &ms_a30);

				printf("%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.8g,%.8g,%.8g\n",
					id, (double)kappa, (double)theta, (double)sigma,
					(double)(feller_lhs - feller_rhs), (double)ms_e, (double)ms_a1, (double)ms_a30,
					(double)mean_e, (double)mean_a1, (double)mean_a30);
				fflush(stdout);
				id++;
			}
		}
	}

	fprintf(stderr, "# Done. Rows printed: %d\n", id);
	fprintf(stderr, "# Interpretation: coarser dt=1/30 for almost-exact -> fewer CIR/Poisson/Gamma steps -> usually faster (ms_almost lower) but mean_almost_30 may differ from mean_almost_1000 (time discretization error / bias).\n");

	TEST_CUDA(cudaFree(states));
	TEST_CUDA(cudaFree(sum));
	return 0;
}
