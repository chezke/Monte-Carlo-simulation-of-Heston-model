/*
 * Proj2026 — Heston exact-style simulation (course Steps 1–3) + Monte Carlo
 *
 * Step 1 (CIR variance, fixed dt): Poisson + standard Gamma G(shape), [4] transition.
 * Step 2: vI += 0.5*(v + v_next)*dt  (trapezoidal approx. of int_0^T v(s)ds, T=1).
 * Step 3: Ivw = (v1-v0-kappa*theta*T+kappa*vI)/sigma  (PDF often writes kappa*theta for T=1),
 *         m = -0.5*vI + rho*Ivw,  Sigma = sqrt((1-rho^2)*vI),  S1 = exp(m + Sigma*G).
 *
 * Gamma: see heston_gamma.cuh ([8] Marsaglia & Tsang, 2000).
 */

#include <math.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "heston_mc_common.cuh"
#include "heston_cuda_utils.cuh"
#include "heston_gamma.cuh"

__global__ void init_curand_state_k(curandState* state, unsigned long long seed) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__device__ float heston_exact_path_payoff(
	curandState* local,
	float S0, float v0, float K, float T,
	float dt, int n_steps,
	float kappa, float theta, float sigma, float rho) {

	const float v_init = v0;
	float v = v0;
	float vI = 0.0f;

	const float exp_m_kdt = expf(-kappa * dt);
	const float om = 1.f - exp_m_kdt;
	const float scale = (sigma * sigma * om) / (2.f * kappa);
	const float d_shape = (2.f * kappa * theta) / (sigma * sigma);
	const float lam_num = 2.f * kappa * exp_m_kdt;
	const float lam_den = (sigma * sigma) * om;

	for (int step = 0; step < n_steps; step++) {
		double lam = (double)((lam_num * v) / lam_den);
		if (lam < 0.0)
			lam = 0.0;
		unsigned int N = curand_poisson(local, lam);
		float shape = d_shape + (float)N;
		float G = gamma_standard_mt(local, shape);
		float v_next = scale * G;
		vI += 0.5f * (v + v_next) * dt;
		v = v_next;
	}

	const float v1 = v;
	const float Ivw = (1.f / sigma) * (v1 - v_init - kappa * theta * T + kappa * vI);
	const float m = -0.5f * vI + rho * Ivw;
	const float sig2 = fmaxf(0.f, (1.f - rho * rho) * vI);
	const float Sigma = sqrtf(sig2);
	const float G_final = curand_normal(local);
	float S1 = S0 * expf(m + Sigma * G_final);
	return fmaxf(0.f, S1 - K);
}

__global__ void heston_exact_mc_k(
	float S0, float v0, float K, float T,
	float dt, int n_steps,
	float kappa, float theta, float sigma, float rho,
	curandState* state,
	float* sum, int n_paths) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	float payoff = 0.f;
	if (idx < n_paths) {
		curandState localState = state[idx];
		payoff = heston_exact_path_payoff(
			&localState, S0, v0, K, T, dt, n_steps,
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

int main(void) {
	const float S0 = 1.0f;
	const float v0 = 0.1f;
	const float K = 1.0f;
	const float T = 1.0f;
	const float kappa = 0.5f;
	const float theta = 0.1f;
	const float sigma = 0.3f;
	const float rho = HESTON_RHO;

	const float dt = 1.0f / 1000.0f;
	const int n_steps = (int)lroundf(T / dt);
	if (fabsf((float)n_steps * dt - T) > 1e-4f * T) {
		fprintf(stderr, "Warning: T not a multiple of dt; effective T=%g\n",
			(double)((float)n_steps * dt));
	}

	const int ntpb = HESTON_MC_THREADS_PER_BLOCK;
	const int n_paths = HESTON_MC_N_PATHS;
	const int nb = (n_paths + ntpb - 1) / ntpb;
	const int grid_threads = nb * ntpb;

	float* sum = nullptr;
	TEST_CUDA(cudaMallocManaged(&sum, 2 * sizeof(float)));
	TEST_CUDA(cudaMemset(sum, 0, 2 * sizeof(float)));

	curandState* states = nullptr;
	TEST_CUDA(cudaMalloc(&states, (size_t)grid_threads * sizeof(curandState)));
	init_curand_state_k<<<nb, ntpb>>>(states, HESTON_MC_CURAND_SEED);
	TEST_CUDA(cudaGetLastError());

	cudaEvent_t start, stop;
	TEST_CUDA(cudaEventCreate(&start));
	TEST_CUDA(cudaEventCreate(&stop));
	TEST_CUDA(cudaEventRecord(start));

	size_t shmem = 2 * (size_t)ntpb * sizeof(float);
	heston_exact_mc_k<<<nb, ntpb, shmem>>>(
		S0, v0, K, T, dt, n_steps,
		kappa, theta, sigma, rho,
		states, sum, n_paths);
	TEST_CUDA(cudaGetLastError());
	TEST_CUDA(cudaDeviceSynchronize());

	TEST_CUDA(cudaEventRecord(stop));
	TEST_CUDA(cudaEventSynchronize(stop));
	float elapsed_ms = 0.0f;
	TEST_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

	double mean = (double)sum[0];
	double MeanSq = (double)sum[1];
	double var_hat = ((double)n_paths / (double)(n_paths - 1)) * (MeanSq - mean * mean);
	if (var_hat < 0.0)
		var_hat = 0.0;
	double stderr = sqrt(var_hat / (double)n_paths);
	double ci95 = 1.96 * stderr;

	printf("Heston exact scheme MC (Proj2026, Steps 1–3 + [8] Gamma)\n");
	printf("Parameters: kappa=%g, theta=%g, sigma=%g, rho=%g, dt=%g, n_steps=%d, T=%g\n",
		(double)kappa, (double)theta, (double)sigma, (double)rho, (double)dt, n_steps, (double)T);
	printf("Paths: %d (CUDA threads launched: %d)\n", n_paths, grid_threads);
	printf("Estimated call (undiscounted) = %.8f\n", mean);
	printf("95%% CI half-width             = %.8f\n", ci95);
	printf("Execution time (kernel)      = %.3f ms\n", (double)elapsed_ms);

	TEST_CUDA(cudaFree(states));
	TEST_CUDA(cudaFree(sum));
	TEST_CUDA(cudaEventDestroy(start));
	TEST_CUDA(cudaEventDestroy(stop));
	return 0;
}