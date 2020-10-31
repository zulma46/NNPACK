#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include <nnpack/utils.h>
#include <nnpack/softmax.h>

#ifdef _WIN32
float max__avx(size_t n, const float* __restrict v);
float sum_exp_minus_c__avx2(size_t n, const float* __restrict v, float c);
void scaled_exp_minus_c__avx2(size_t n, const float* __restrict x, float* __restrict y, float scale, float c);
void inplace_scaled_exp_minus_c__avx2(size_t n, const float* __restrict v, float scale, float c);

void nnp_softmax__avx2(
	size_t n,
	const float* __restrict x, 
	float* __restrict y)
{
	const float c = max__avx(n, x);
	const float sum = sum_exp_minus_c__avx2(n, x, c);
	const float scale = 1.0f / sum;
	scaled_exp_minus_c__avx2(n, x, y, scale, c);
}

void nnp_inplace_softmax__avx2(
	size_t n,
	float* __restrict v)
{
	const float c = max__avx(n, v);
	const float sum = sum_exp_minus_c__avx2(n, v, c);
	const float scale = 1.0f / sum;
	inplace_scaled_exp_minus_c__avx2(n, v, scale, c);
}
#else
float max__avx(size_t n, const float v[restrict static n]);
float sum_exp_minus_c__avx2(size_t n, const float v[restrict static n], float c);
void scaled_exp_minus_c__avx2(size_t n, const float x[restrict static n], float y[restrict static n], float scale, float c);
void inplace_scaled_exp_minus_c__avx2(size_t n, const float v[restrict static n], float scale, float c);

void nnp_softmax__avx2(
	size_t n,
	const float x[restrict static n],
	float y[restrict static n])
{
	const float c = max__avx(n, x);
	const float sum = sum_exp_minus_c__avx2(n, x, c);
	const float scale = 1.0f / sum;
	scaled_exp_minus_c__avx2(n, x, y, scale, c);
}

void nnp_inplace_softmax__avx2(
	size_t n,
	float v[restrict static n])
{
	const float c = max__avx(n, v);
	const float sum = sum_exp_minus_c__avx2(n, v, c);
	const float scale = 1.0f / sum;
	inplace_scaled_exp_minus_c__avx2(n, v, scale, c);
}
#endif
