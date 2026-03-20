/*
 * One CIR variance step: exact transition (Poisson + Gamma), same as Proj Step 1 / [4].
 */
#ifndef HESTON_CIR_EXACT_CUH
#define HESTON_CIR_EXACT_CUH

#include "heston_gamma.cuh"

__device__ float cir_exact_variance_step(curandState* st, float v, float kappa, float theta, float sigma, float dt) {
	const float exp_m_kdt = expf(-kappa * dt);
	const float om = 1.f - exp_m_kdt;
	if (!(om > 1e-20f))
		return fmaxf(v, 0.f);

	const float scale = (sigma * sigma * om) / (2.f * kappa);
	const float d_shape = (2.f * kappa * theta) / (sigma * sigma);
	const float lam_num = 2.f * kappa * exp_m_kdt;
	const float lam_den = (sigma * sigma) * om;

	double lam = (double)((lam_num * v) / lam_den);
	if (lam < 0.0)
		lam = 0.0;
	unsigned int N = curand_poisson(st, lam);
	float shape = d_shape + (float)N;
	float G = gamma_standard_mt(st, shape);
	return scale * G;
}

#endif /* HESTON_CIR_EXACT_CUH */
