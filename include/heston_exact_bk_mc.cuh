/*
 * Broadie–Kaya-style exact simulation (course Steps 1–3): one path + MC reduction kernel.
 * Shared by src/MC_exact.cu and src/MC_benchmark_Q3.cu.
 *
 * Step 1: CIR variance via Poisson + Gamma ([4] transition; Gamma: heston_gamma.cuh).
 * Step 2: vI += 0.5*(v + v_next)*dt
 * Step 3: Ivw = (v1-v0-kappa*theta*T+kappa*vI)/sigma, m = -0.5*vI + rho*Ivw,
 *         S1 = S0*exp(m + sqrt((1-rho^2)*vI)*G)
 */

#ifndef HESTON_EXACT_BK_MC_CUH
#define HESTON_EXACT_BK_MC_CUH

#include <math.h>
#include <curand_kernel.h>
#include "heston_gamma.cuh"

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

#endif /* HESTON_EXACT_BK_MC_CUH */
