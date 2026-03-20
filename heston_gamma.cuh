/*
 * Marsaglia & Tsang (2000) standard Gamma G(alpha), density ∝ y^{alpha-1} e^{-y}, y>0.
 * Used by exact / almost-exact Heston variance transitions ([8]).
 */
#ifndef HESTON_GAMMA_CUH
#define HESTON_GAMMA_CUH

#include <curand_kernel.h>

__device__ float gamma_standard_mt(curandState* st, float alpha) {
	if (!(alpha > 0.f))
		alpha = 1e-8f;
	if (alpha < 1.f) {
		float u = curand_uniform(st);
		while (u < 1e-20f)
			u = curand_uniform(st);
		return gamma_standard_mt(st, alpha + 1.f) * powf(u, 1.f / alpha);
	}

	const float d = alpha - (1.f / 3.f);
	const float c = 1.f / sqrtf(9.f * d);

	for (;;) {
		float x, v_lin;
		do {
			x = curand_normal(st);
			v_lin = 1.f + c * x;
		} while (v_lin <= 0.f);

		const float v = v_lin * v_lin * v_lin;
		float u = curand_uniform(st);
		while (u <= 0.f)
			u = curand_uniform(st);

		const float x2 = x * x;
		if (u < 1.f - 0.0331f * x2 * x2)
			return d * v;

		if (logf(u) < 0.5f * x2 + d * (1.f - v + logf(v)))
			return d * v;
	}
}

#endif /* HESTON_GAMMA_CUH */
