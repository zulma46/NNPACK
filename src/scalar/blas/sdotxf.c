#include <stddef.h>


void nnp_sdotxf1__scalar(
#ifdef _WIN32
	const float* x,
	const float* y,
#else
	const float x[restrict static 1],
	const float y[restrict static 1],
#endif
	size_t stride_y,
#ifdef _WIN32
    float* sum,
#else
	float sum[restrict static 1],
#endif
	size_t n)
{
	float acc0 = 0.0f;
#ifdef _WIN32
	const float* y0 = y;
#else
	const float *restrict y0 = y;
#endif
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
	}
	sum[0] = acc0;
}

void nnp_sdotxf2__scalar(
#ifdef _WIN32
	const float* x,
	const float* y,
#else
	const float x[restrict static 1],
	const float y[restrict static 2],
#endif
	size_t stride_y,
#ifdef _WIN32
	float* sum,
#else
	float sum[restrict static 2],
#endif
	size_t n)
{
	float acc0, acc1;
	acc0 = acc1 = 0.0f;
#ifdef _WIN32
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
#else
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
#endif
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
}

void nnp_sdotxf3__scalar(
#ifdef _WIN32
    const float* x,
	const float* y,
#else
	const float x[restrict static 1],
	const float y[restrict static 3],
#endif
	size_t stride_y,
#ifdef _WIN32
	float* sum,
#else
	float sum[restrict static 3],
#endif
	size_t n)
{
	float acc0, acc1, acc2;
	acc0 = acc1 = acc2 = 0.0f;
#ifdef _WIN32
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
#else
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
#endif
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
}

void nnp_sdotxf4__scalar(
#ifdef _WIN32
	const float* x,
	const float* y,
#else
	const float x[restrict static 1],
	const float y[restrict static 4],
#endif
	size_t stride_y,
#ifdef _WIN32
	float* sum,
#else
	float sum[restrict static 4],
#endif
	size_t n)
{
	float acc0, acc1, acc2, acc3;
	acc0 = acc1 = acc2 = acc3 = 0.0f;
#ifdef _WIN32
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
#else
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
#endif
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
}

void nnp_sdotxf5__scalar(
#ifdef _WIN32
	const float* x,
	const float* y,
#else
	const float x[restrict static 1],
	const float y[restrict static 5],
#endif
	size_t stride_y,
#ifdef _WIN32
	float* sum,
#else
	float sum[restrict static 5],
#endif
	size_t n)
{
	float acc0, acc1, acc2, acc3, acc4;
	acc0 = acc1 = acc2 = acc3 = acc4 = 0.0f;
#ifdef _WIN32
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
	const float* y4 = y3 + stride_y;
#else
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
#endif
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
		acc4 += vx * (*y4++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
}

void nnp_sdotxf6__scalar(
#ifdef _WIN32
	const float* x,
	const float* y,
#else
	const float x[restrict static 1],
	const float y[restrict static 6],
#endif
	size_t stride_y,
#ifdef _WIN32
	float* sum,
#else
	float sum[restrict static 6],
#endif
	size_t n)
{
	float acc0, acc1, acc2, acc3, acc4, acc5;
	acc0 = acc1 = acc2 = acc3 = acc4 = acc5 = 0.0f;
#ifdef _WIN32
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
	const float* y4 = y3 + stride_y;
	const float* y5 = y4 + stride_y;
#else
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	const float *restrict y5 = y4 + stride_y;
#endif
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
		acc4 += vx * (*y4++);
		acc5 += vx * (*y5++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
}

void nnp_sdotxf7__scalar(
#ifdef _WIN32
	const float* x,
	const float* y,
#else
	const float x[restrict static 1],
	const float y[restrict static 7],
#endif
	size_t stride_y,
#ifdef _WIN32
	float* sum,
#else
	float sum[restrict static 7],
#endif
	size_t n)
{
	float acc0, acc1, acc2, acc3, acc4, acc5, acc6;
	acc0 = acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = 0.0f;
#ifdef _WIN32
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
	const float* y4 = y3 + stride_y;
	const float* y5 = y4 + stride_y;
	const float* y6 = y5 + stride_y;
#else
	const float *restrict y0 = y;
	const float *restrict y1 = y0 + stride_y;
	const float *restrict y2 = y1 + stride_y;
	const float *restrict y3 = y2 + stride_y;
	const float *restrict y4 = y3 + stride_y;
	const float *restrict y5 = y4 + stride_y;
	const float *restrict y6 = y5 + stride_y;
#endif
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
		acc4 += vx * (*y4++);
		acc5 += vx * (*y5++);
		acc6 += vx * (*y6++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
	sum[6] = acc6;
}

void nnp_sdotxf8__scalar(
#ifdef _WIN32
	const float* x,
	const float* y,
#else
	const float x[restrict static 1],
	const float y[restrict static 8],
#endif
	size_t stride_y,
#ifdef _WIN32
	float* sum,
#else
	float sum[restrict static 2],
#endif
	size_t n)
{
	float acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
	acc0 = acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = acc7 = 0.0f;
#ifdef _WIN32
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
	const float* y4 = y3 + stride_y;
	const float* y5 = y4 + stride_y;
	const float* y6 = y5 + stride_y;
	const float* y7 = y6 + stride_y;
#else
	const float* restrict y0 = y;
	const float* restrict y1 = y0 + stride_y;
	const float* restrict y2 = y1 + stride_y;
	const float* restrict y3 = y2 + stride_y;
	const float* restrict y4 = y3 + stride_y;
	const float* restrict y5 = y4 + stride_y;
	const float* restrict y6 = y5 + stride_y;
	const float* restrict y7 = y6 + stride_y;
#endif
	
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
		acc4 += vx * (*y4++);
		acc5 += vx * (*y5++);
		acc6 += vx * (*y6++);
		acc7 += vx * (*y7++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
	sum[6] = acc6;
	sum[7] = acc7;
}
