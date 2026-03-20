/*
 * Proj2026 — Question 1: Heston model, Euler discretization + Monte Carlo
 * SDE (risk-neutral, r = 0 in the project statement):
 *   dS = r S dt + sqrt(v) S dZhat,   dZhat = rho dW + sqrt(1-rho^2) dZ
 *   dv = kappa (theta - v) dt + sigma sqrt(v) dW
 *
 * Euler (same structure as the project PDF, with dt = Delta t):
 *   S <- S + r*S*dt + sqrt(max(v,0))*S*sqrt(dt) * (rho*G1 + sqrt(1-rho^2)*G2)
 *   v <- g( v + kappa*(theta-v)*dt + sigma*sqrt(max(v,0))*sqrt(dt)*G1 )
 *   where g is (.)+ by default; set USE_ABS_VARIANCE_TRUNC 1 for |.|.
 *
 * Correlation rho: default in heston_mc_common.cuh (override with -DHESTON_RHO=...).
 */

#include <math.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "heston_mc_common.cuh"
#include "heston_cuda_utils.cuh"

#ifndef USE_ABS_VARIANCE_TRUNC
#define USE_ABS_VARIANCE_TRUNC 0
#endif

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

/*
 * One path: n_steps Euler steps with fixed dt (T = n_steps * dt).
 * Payoff at T: (S - K)+ with r = 0 => no discount.
 */
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

	/* Idle threads (idx >= n_paths) keep payoff = 0 so they still join __syncthreads(). */
	float payoff = 0.0f;
	if (idx < n_paths) {
		curandState localState = state[idx];
		payoff = heston_euler_path_payoff(
			&localState, S0, v0, r, K, dt, n_steps, kappa, theta, sigma, rho);
		state[idx] = localState;
	}

	extern __shared__ float sh[];
	float* R1 = sh;
	float* R2 = R1 + blockDim.x;
	/* Per-thread: ψ_i = Ψ_i/n_paths;  R2[i] = n_paths*ψ_i^2 = Ψ_i^2/n_paths.
	 * Global sums satisfy Σ ψ_i = mean(Ψ) and Σ R2 entries = (1/n_paths) Σ Ψ_i^2. */
	R1[threadIdx.x] = payoff / (float)n_paths;
	R2[threadIdx.x] = R1[threadIdx.x] * R1[threadIdx.x] * (float)n_paths;

	// Block reduction of Σ ψ_i and Σ Ψ_i^2/n_paths, then atomic add into sum[0], sum[1].
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			R1[threadIdx.x] += R1[threadIdx.x + stride];
			R2[threadIdx.x] += R2[threadIdx.x + stride];
		}
		__syncthreads();
	}

    // Write the results to the global memory.
	if (threadIdx.x == 0) {
		atomicAdd(sum, R1[0]);
		atomicAdd(sum + 1, R2[0]);
	}
}

int main(void) {
	/* Project Q1 */
	const float S0 = 1.0f;
	const float v0 = 0.1f;
	const float r = 0.0f;
	const float K = 1.0f;
	const float T = 1.0f;

	const float kappa = 0.5f;
	const float theta = 0.1f;
	const float sigma = 0.3f;
	const float rho = HESTON_RHO;

	const float dt = 1.0f / 1000.0f;
	const int n_steps = (int)lroundf(T / dt);
	if (fabsf((float)n_steps * dt - T) > 1e-4f * T) {
		fprintf(stderr, "Warning: T not multiple of dt; using n_steps=%d, effective T=%g\n",
			n_steps, (double)((float)n_steps * dt));
	}

	const int ntpb = HESTON_MC_THREADS_PER_BLOCK;
	const int n_paths = HESTON_MC_N_PATHS;
	const int nb = (n_paths + ntpb - 1) / ntpb;
	const int grid_threads = nb * ntpb;

	float* sum = nullptr;
	TEST_CUDA(cudaMallocManaged(&sum, 2 * sizeof(float)));
	TEST_CUDA(cudaMemset(sum, 0, 2 * sizeof(float)));

	curandState* states = nullptr;
    // TEST_CUDA(cudaMalloc(&states, (size_t)n_paths * sizeof(curandState)));
	TEST_CUDA(cudaMalloc(&states, (size_t)grid_threads * sizeof(curandState)));
	init_curand_state_k<<<nb, ntpb>>>(states, HESTON_MC_CURAND_SEED);
	TEST_CUDA(cudaGetLastError());

	cudaEvent_t start, stop;
	TEST_CUDA(cudaEventCreate(&start));
	TEST_CUDA(cudaEventCreate(&stop));
	TEST_CUDA(cudaEventRecord(start));

	size_t shmem = 2 * (size_t)ntpb * sizeof(float);
	heston_euler_mc_k<<<nb, ntpb, shmem>>>(
		S0, v0, r, K, dt, n_steps,
		kappa, theta, sigma, rho,
		states, sum, n_paths);
	TEST_CUDA(cudaGetLastError());
	TEST_CUDA(cudaDeviceSynchronize());

	TEST_CUDA(cudaEventRecord(stop));
	TEST_CUDA(cudaEventSynchronize(stop));
	float elapsed_ms = 0.0f;
	TEST_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Calculate the mean and the variance of the payoffs.
	double mean = (double)sum[0];
	double MeanSq = (double)sum[1];
	double var_hat = ((double)n_paths / (double)(n_paths - 1)) * (MeanSq - mean * mean);
	if (var_hat < 0.0) var_hat = 0.0; /* FP noise can make this slightly negative */
	double stderr = sqrt(var_hat / (double)n_paths);
	double ci95 = 1.96 * stderr;

	printf("Heston Euler MC (Proj2026 Q1)\n");
	printf("Parameters: kappa=%g, theta=%g, sigma=%g, rho=%g, dt=%g, n_steps=%d\n",
		(double)kappa, (double)theta, (double)sigma, (double)rho, (double)dt, n_steps);
	// printf("Paths: %d\n", n_paths);
    printf("Paths: %d (CUDA threads launched: %d)\n", n_paths, grid_threads);
#if USE_ABS_VARIANCE_TRUNC
	printf("Variance truncation g: absolute value |.| \n");
#else
	printf("Variance truncation g: positive part (.)+ \n");
#endif
	printf("Estimated E[(S_T - K)+]  = %.8f\n", mean);
	printf("95%% CI half-width        = %.8f\n", ci95);
	printf("Execution time (kernel)   = %.3f ms\n", (double)elapsed_ms);

	TEST_CUDA(cudaFree(states));
	TEST_CUDA(cudaFree(sum));
	TEST_CUDA(cudaEventDestroy(start));
	TEST_CUDA(cudaEventDestroy(stop));
	return 0;
}

/*
The current implementation uses an explicit Euler 
discretization with a truncation scheme to ensure the 
variance remains non-negative. This approach is simple 
and easy to implement, but it introduces discretization 
bias and may suffer from instability.
Therefore, it is mainly used as a baseline method to 
compare with more advanced schemes such as the exact 
simulation or the QE scheme.

However, there is a potential issue with the Euler update 
of S. The Euler discretization may produce negative asset 
prices because the update is linear in the noise term. 
Since the Gaussian variable can take large negative 
values, the multiplicative factor can become negative. 
However, for small time steps, this issue is rare in 
practice. Nevertheless, it is a known numerical drawback 
of the Euler scheme, as the true continuous model guarantees 
positivity. (S_t = S_0 \exp(\text{something}))
*/