/*
 * Proj2026 — Almost exact scheme for log S ([10]-style), variance v via exact CIR steps.
 *   log S_{t+dt} = log S_t + k0 + k1 v_t + k2 v_{t+dt} + sqrt((1-rho^2) v_t dt) G2
 * with k0,k1,k2 as in the project statement; r = 0.
 */

#include <math.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "heston_mc_common.cuh"
#include "heston_cuda_utils.cuh"
#include "heston_cir_exact.cuh"

__global__ void init_curand_state_k(curandState* state, unsigned long long seed) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
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

int main(void) {
	const float S0 = 1.f;
	const float v0 = 0.1f;
	const float r = 0.f;
	const float K = 1.f;
	const float T = 1.f;
	const float kappa = 0.5f;
	const float theta = 0.1f;
	const float sigma = 0.3f;
	const float rho = HESTON_RHO;

	const float dt = 1.f / 1000.f;
	const int n_steps = (int)lroundf(T / dt);

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
	heston_almost_mc_k<<<nb, ntpb, shmem>>>(
		S0, v0, r, K, dt, n_steps, kappa, theta, sigma, rho, states, sum, n_paths);
	TEST_CUDA(cudaGetLastError());
	TEST_CUDA(cudaDeviceSynchronize());
	TEST_CUDA(cudaEventRecord(stop));
	TEST_CUDA(cudaEventSynchronize(stop));
	float elapsed_ms = 0.f;
	TEST_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

	double mean = (double)sum[0];
	double MeanSq = (double)sum[1];
	double var_hat = ((double)n_paths / (double)(n_paths - 1)) * (MeanSq - mean * mean);
	if (var_hat < 0.0)
		var_hat = 0.0;

	printf("Heston almost-exact logS (variance: exact transition), Proj Q3 scheme\n");
	printf("kappa=%g theta=%g sigma=%g rho=%g dt=%g n_steps=%d\n",
		(double)kappa, (double)theta, (double)sigma, (double)rho, (double)dt, n_steps);
	printf("Paths: %d  time_ms: %.3f  mean_payoff: %.8f\n", n_paths, (double)elapsed_ms, mean);

	TEST_CUDA(cudaFree(states));
	TEST_CUDA(cudaFree(sum));
	TEST_CUDA(cudaEventDestroy(start));
	TEST_CUDA(cudaEventDestroy(stop));
	return 0;
}
