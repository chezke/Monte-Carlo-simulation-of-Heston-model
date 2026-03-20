/*
 * Host-side CUDA API error checking (shared by Heston .cu drivers).
 */
#ifndef HESTON_CUDA_UTILS_CUH
#define HESTON_CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static inline void testCUDA(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define TEST_CUDA(e) testCUDA((e), __FILE__, __LINE__)

#endif /* HESTON_CUDA_UTILS_CUH */
