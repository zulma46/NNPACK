#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include <nnpack/utils.h>
#include <nnpack/softmax.h>


static float max__scalar(size_t n, 
#ifdef _WIN32
	const float* v) {
#else
	const float v[restrict static n]) {
#endif
	float max_v = *v++;
	while (--n) {
		max_v = maxf(max_v, *v++);
	}
	return max_v;
}

static float sum_exp_minus_c__scalar(size_t n, 
#ifdef _WIN32
	const float* v,
#else
	const float v[restrict static n], 
#endif
	float c) {
	float sum = 0.0f;
	do {
		sum += expf(*v++ - c);
	} while (--n);
	return sum;
}

static void scaled_exp_minus_c__scalar(size_t n, 
#ifdef _WIN32
	const float* x,
	float* y,
#else
	const float x[static n], 
	float y[static n],
#endif
	float scale, float c) {
	do {
		*y++ = scale * expf(*x++ - c);
	} while (--n);
}

void nnp_softmax__scalar(
	size_t n,
#ifdef _WIN32
	const float* x,
	float* y)
#else
	const float x[restrict static n],
	float y[restrict static n])
#endif
{
	const float c = max__scalar(n, x);
	const float sum = sum_exp_minus_c__scalar(n, x, c);
	const float scale = 1.0f / sum;
	scaled_exp_minus_c__scalar(n, x, y, scale, c);
}

void nnp_inplace_softmax__scalar(
	size_t n,
#ifdef _WIN32
	float* v)
#else
	float v[restrict static n])
#endif
{
	const float c = max__scalar(n, v);
	const float sum = sum_exp_minus_c__scalar(n, v, c);
	const float scale = 1.0f / sum;
	scaled_exp_minus_c__scalar(n, v, v, scale, c);
}
