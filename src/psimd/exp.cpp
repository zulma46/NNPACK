#include <stddef.h>

#include <psimd.h>

#include <nnpack/softmax.h>

#ifdef __cplusplus 
extern "C" {
#endif

void nnp_vector_exp__psimd(
	size_t n,
	const float* x,
	float* y)
{
	do {
		psimd_store_f32(y, exp(psimd_load_f32(x)));

		y += 4;
		x += 4;
		n -= 4;
	} while (n >= 4);
	if (n != 0) {
		psimd_store_f32(y + n - 4, exp(psimd_load_f32(x + n - 4)));
	}
}

#ifdef __cplusplus
}
#endif
