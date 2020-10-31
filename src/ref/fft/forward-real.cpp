#include <nnpack/fft.h>
#include <ref/fft/complex-ref.h>


void nnp_fft8_real__ref(const float* t, size_t t_stride, float* f, size_t f_stride)
{
	/* Load inputs */
	std::complex<float> w0 = std::complex<float>(t[0 * t_stride], t[1 * t_stride]);
	std::complex<float> w1 = std::complex<float>(t[2 * t_stride], t[3 * t_stride]);
	std::complex<float> w2 = std::complex<float>(t[4 * t_stride], t[5 * t_stride]);
	std::complex<float> w3 = std::complex<float>(t[6 * t_stride], t[7 * t_stride]);

	fft4fc(&w0, &w1, &w2, &w3);

	const float two_g1_real = std::real<float>(w1) + std::real<float>(w3);
	const float two_g1_imag = std::imag<float>(w1) - std::imag<float>(w3);
	const float two_h1_real = std::imag<float>(w1) + std::imag<float>(w3);
	const float two_h1_imag = std::real<float>(w3) - std::real<float>(w1);

	const float two_h1_real_plus_imag = two_h1_real + two_h1_imag;
	const float two_h1_real_minus_imag = two_h1_real - two_h1_imag;

	const float two_w1_real = two_g1_real + SQRT2_OVER_2 * two_h1_real_plus_imag;
	const float two_w1_imag = two_g1_imag - SQRT2_OVER_2 * two_h1_real_minus_imag;
	const float two_w3_real = two_g1_real - SQRT2_OVER_2 * two_h1_real_plus_imag;
	const float two_w3_imag = -two_g1_imag - SQRT2_OVER_2 * two_h1_real_minus_imag;

	/* Store outputs */
	f[0 * f_stride] = std::real<float>(w0) + std::imag<float>(w0);
	f[1 * f_stride] = std::real<float>(w0) - std::imag<float>(w0);
	f[2 * f_stride] = 0.5f * two_w1_real;
	f[3 * f_stride] = 0.5f * two_w1_imag;
	f[4 * f_stride] = std::real<float>(w2);
	f[5 * f_stride] = -std::imag<float>(w2);
	f[6 * f_stride] = 0.5f * two_w3_real;
	f[7 * f_stride] = 0.5f * two_w3_imag;
}

void nnp_fft16_real__ref(const float* t, size_t t_stride, float* f, size_t f_stride)
{
	/* Load inputs */
	std::complex<float> w0 = std::complex<float>(t[ 0 * t_stride], t[ 1 * t_stride]);
	std::complex<float> w1 = std::complex<float>(t[ 2 * t_stride], t[ 3 * t_stride]);
	std::complex<float> w2 = std::complex<float>(t[ 4 * t_stride], t[ 5 * t_stride]);
	std::complex<float> w3 = std::complex<float>(t[ 6 * t_stride], t[ 7 * t_stride]);
	std::complex<float> w4 = std::complex<float>(t[ 8 * t_stride], t[ 9 * t_stride]);
	std::complex<float> w5 = std::complex<float>(t[10 * t_stride], t[11 * t_stride]);
	std::complex<float> w6 = std::complex<float>(t[12 * t_stride], t[13 * t_stride]);
	std::complex<float> w7 = std::complex<float>(t[14 * t_stride], t[15 * t_stride]);

	fft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	const float two_g1_real = std::real<float>(w1) + std::real<float>(w7);
	const float two_g1_imag = std::imag<float>(w1) - std::imag<float>(w7);
	const float two_g2_real = std::real<float>(w2) + std::real<float>(w6);
	const float two_g2_imag = std::imag<float>(w2) - std::imag<float>(w6);
	const float two_g3_real = std::real<float>(w3) + std::real<float>(w5);
	const float two_g3_imag = std::imag<float>(w3) - std::imag<float>(w5);

	const float two_h1_real = std::imag<float>(w1) + std::imag<float>(w7);
	const float two_h1_imag = std::real<float>(w7) - std::real<float>(w1);
	const float two_h2_real = std::imag<float>(w2) + std::imag<float>(w6);
	const float two_h2_imag = std::real<float>(w6) - std::real<float>(w2);
	const float two_h3_real = std::imag<float>(w3) + std::imag<float>(w5);
	const float two_h3_imag = std::real<float>(w5) - std::real<float>(w3);

	const float two_w1_real = two_g1_real + two_h1_real * COS_1PI_OVER_8 + two_h1_imag * COS_3PI_OVER_8;
	const float two_w1_imag = two_g1_imag + two_h1_imag * COS_1PI_OVER_8 - two_h1_real * COS_3PI_OVER_8;
	const float two_w3_real = two_g3_real + two_h3_real * COS_3PI_OVER_8 + two_h3_imag * COS_1PI_OVER_8;
	const float two_w3_imag = two_g3_imag + two_h3_imag * COS_3PI_OVER_8 - two_h3_real * COS_1PI_OVER_8;

	const float two_w2_real = two_g2_real + SQRT2_OVER_2 * (two_h2_real + two_h2_imag);
	const float two_w2_imag = two_g2_imag + SQRT2_OVER_2 * (two_h2_imag - two_h2_real);
	const float two_w6_real = two_g2_real - SQRT2_OVER_2 * (two_h2_real + two_h2_imag);
	const float two_w6_imag = -two_g2_imag + SQRT2_OVER_2 * (two_h2_imag - two_h2_real);

	const float two_w5_real = two_g3_real - two_h3_real * COS_3PI_OVER_8 - two_h3_imag * COS_1PI_OVER_8;
	const float two_w5_imag = -two_g3_imag + two_h3_imag * COS_3PI_OVER_8 - two_h3_real * COS_1PI_OVER_8;
	const float two_w7_real = two_g1_real - two_h1_real * COS_1PI_OVER_8 - two_h1_imag * COS_3PI_OVER_8;
	const float two_w7_imag = -two_g1_imag + two_h1_imag * COS_1PI_OVER_8 - two_h1_real * COS_3PI_OVER_8;

	/* Store outputs */
	f[ 0 * f_stride] = std::real<float>(w0) + std::imag<float>(w0);
	f[ 1 * f_stride] = std::real<float>(w0) - std::imag<float>(w0);
	f[ 2 * f_stride] = 0.5f * two_w1_real;
	f[ 3 * f_stride] = 0.5f * two_w1_imag;
	f[ 4 * f_stride] = 0.5f * two_w2_real;
	f[ 5 * f_stride] = 0.5f * two_w2_imag;
	f[ 6 * f_stride] = 0.5f * two_w3_real;
	f[ 7 * f_stride] = 0.5f * two_w3_imag;
	f[ 8 * f_stride] = std::real<float>(w4);
	f[ 9 * f_stride] = -std::imag<float>(w4);
	f[10 * f_stride] = 0.5f * two_w5_real;
	f[11 * f_stride] = 0.5f * two_w5_imag;
	f[12 * f_stride] = 0.5f * two_w6_real;
	f[13 * f_stride] = 0.5f * two_w6_imag;
	f[14 * f_stride] = 0.5f * two_w7_real;
	f[15 * f_stride] = 0.5f * two_w7_imag;
}

void nnp_fft32_real__ref(const float* t, size_t t_stride, float* f, size_t f_stride)
{
	/* Load inputs */
	std::complex<float> w0  = std::complex<float>(t[ 0 * t_stride], t[ 1 * t_stride]);
	std::complex<float> w1  = std::complex<float>(t[ 2 * t_stride], t[ 3 * t_stride]);
	std::complex<float> w2  = std::complex<float>(t[ 4 * t_stride], t[ 5 * t_stride]);
	std::complex<float> w3  = std::complex<float>(t[ 6 * t_stride], t[ 7 * t_stride]);
	std::complex<float> w4  = std::complex<float>(t[ 8 * t_stride], t[ 9 * t_stride]);
	std::complex<float> w5  = std::complex<float>(t[10 * t_stride], t[11 * t_stride]);
	std::complex<float> w6  = std::complex<float>(t[12 * t_stride], t[13 * t_stride]);
	std::complex<float> w7  = std::complex<float>(t[14 * t_stride], t[15 * t_stride]);
	std::complex<float> w8  = std::complex<float>(t[16 * t_stride], t[17 * t_stride]);
	std::complex<float> w9  = std::complex<float>(t[18 * t_stride], t[19 * t_stride]);
	std::complex<float> w10 = std::complex<float>(t[20 * t_stride], t[21 * t_stride]);
	std::complex<float> w11 = std::complex<float>(t[22 * t_stride], t[23 * t_stride]);
	std::complex<float> w12 = std::complex<float>(t[24 * t_stride], t[25 * t_stride]);
	std::complex<float> w13 = std::complex<float>(t[26 * t_stride], t[27 * t_stride]);
	std::complex<float> w14 = std::complex<float>(t[28 * t_stride], t[29 * t_stride]);
	std::complex<float> w15 = std::complex<float>(t[30 * t_stride], t[31 * t_stride]);

	fft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	const float two_g1_real = std::real<float>(w1) + std::real<float>(w15);
	const float two_g1_imag = std::imag<float>(w1) - std::imag<float>(w15);
	const float two_g2_real = std::real<float>(w2) + std::real<float>(w14);
	const float two_g2_imag = std::imag<float>(w2) - std::imag<float>(w14);
	const float two_g3_real = std::real<float>(w3) + std::real<float>(w13);
	const float two_g3_imag = std::imag<float>(w3) - std::imag<float>(w13);
	const float two_g4_real = std::real<float>(w4) + std::real<float>(w12);
	const float two_g4_imag = std::imag<float>(w4) - std::imag<float>(w12);
	const float two_g5_real = std::real<float>(w5) + std::real<float>(w11);
	const float two_g5_imag = std::imag<float>(w5) - std::imag<float>(w11);
	const float two_g6_real = std::real<float>(w6) + std::real<float>(w10);
	const float two_g6_imag = std::imag<float>(w6) - std::imag<float>(w10);
	const float two_g7_real = std::real<float>(w7) + std::real<float>(w9);
	const float two_g7_imag = std::imag<float>(w7) - std::imag<float>(w9);

	const float two_h1_real = std::imag<float>(w15)+ std::imag<float>(w1);
	const float two_h1_imag = std::real<float>(w15) - std::real<float>(w1);
	const float two_h2_real = std::imag<float>(w14) + std::imag<float>(w2);
	const float two_h2_imag = std::real<float>(w14) - std::real<float>(w2);
	const float two_h3_real = std::imag<float>(w13) + std::imag<float>(w3);
	const float two_h3_imag = std::real<float>(w13) - std::real<float>(w3);
	const float two_h4_real = std::imag<float>(w12) + std::imag<float>(w4);
	const float two_h4_imag = std::real<float>(w12) - std::real<float>(w4);
	const float two_h5_real = std::imag<float>(w11) + std::imag<float>(w5);
	const float two_h5_imag = std::real<float>(w11) - std::real<float>(w5);
	const float two_h6_real = std::imag<float>(w10) + std::imag<float>(w6);
	const float two_h6_imag = std::real<float>(w10) - std::real<float>(w6);
	const float two_h7_real = std::imag<float>(w9) + std::imag<float>(w7);
	const float two_h7_imag = std::real<float>(w9) - std::real<float>(w7);

	const float two_w1_real = two_g1_real + two_h1_real * COS__1PI_OVER_16 + two_h1_imag * COS__7PI_OVER_16;
	const float two_w1_imag = two_g1_imag + two_h1_imag * COS__1PI_OVER_16 - two_h1_real * COS__7PI_OVER_16;
	const float two_w7_real = two_g7_real + two_h7_real * COS__7PI_OVER_16 + two_h7_imag * COS__1PI_OVER_16;
	const float two_w7_imag = two_g7_imag + two_h7_imag * COS__7PI_OVER_16 - two_h7_real * COS__1PI_OVER_16;

	const float two_w2_real = two_g2_real + two_h2_real * COS__2PI_OVER_16 + two_h2_imag * COS__6PI_OVER_16;
	const float two_w2_imag = two_g2_imag + two_h2_imag * COS__2PI_OVER_16 - two_h2_real * COS__6PI_OVER_16;
	const float two_w6_real = two_g6_real + two_h6_real * COS__6PI_OVER_16 + two_h6_imag * COS__2PI_OVER_16;
	const float two_w6_imag = two_g6_imag + two_h6_imag * COS__6PI_OVER_16 - two_h6_real * COS__2PI_OVER_16;

	const float two_w3_real = two_g3_real + two_h3_real * COS__3PI_OVER_16 + two_h3_imag * COS__5PI_OVER_16;
	const float two_w3_imag = two_g3_imag + two_h3_imag * COS__3PI_OVER_16 - two_h3_real * COS__5PI_OVER_16;
	const float two_w5_real = two_g5_real + two_h5_real * COS__5PI_OVER_16 + two_h5_imag * COS__3PI_OVER_16;
	const float two_w5_imag = two_g5_imag + two_h5_imag * COS__5PI_OVER_16 - two_h5_real * COS__3PI_OVER_16;

	const float two_w4_real  =  two_g4_real + SQRT2_OVER_2 * (two_h4_real + two_h4_imag);
	const float two_w4_imag  =  two_g4_imag + SQRT2_OVER_2 * (two_h4_imag - two_h4_real);
	const float two_w12_real =  two_g4_real - SQRT2_OVER_2 * (two_h4_real + two_h4_imag);
	const float two_w12_imag = -two_g4_imag + SQRT2_OVER_2 * (two_h4_imag - two_h4_real);

	const float two_w9_real  =  two_g7_real - two_h7_real * COS__7PI_OVER_16 - two_h7_imag * COS__1PI_OVER_16;
	const float two_w9_imag  = -two_g7_imag + two_h7_imag * COS__7PI_OVER_16 - two_h7_real * COS__1PI_OVER_16;
	const float two_w15_real =  two_g1_real - two_h1_real * COS__1PI_OVER_16 - two_h1_imag * COS__7PI_OVER_16;
	const float two_w15_imag = -two_g1_imag + two_h1_imag * COS__1PI_OVER_16 - two_h1_real * COS__7PI_OVER_16;

	const float two_w10_real =  two_g6_real - two_h6_real * COS__6PI_OVER_16 - two_h6_imag * COS__2PI_OVER_16;
	const float two_w10_imag = -two_g6_imag + two_h6_imag * COS__6PI_OVER_16 - two_h6_real * COS__2PI_OVER_16;
	const float two_w14_real =  two_g2_real - two_h2_real * COS__2PI_OVER_16 - two_h2_imag * COS__6PI_OVER_16;
	const float two_w14_imag = -two_g2_imag + two_h2_imag * COS__2PI_OVER_16 - two_h2_real * COS__6PI_OVER_16;

	const float two_w11_real =  two_g5_real - two_h5_real * COS__5PI_OVER_16 - two_h5_imag * COS__3PI_OVER_16;
	const float two_w11_imag = -two_g5_imag + two_h5_imag * COS__5PI_OVER_16 - two_h5_real * COS__3PI_OVER_16;
	const float two_w13_real =  two_g3_real - two_h3_real * COS__3PI_OVER_16 - two_h3_imag * COS__5PI_OVER_16;
	const float two_w13_imag = -two_g3_imag + two_h3_imag * COS__3PI_OVER_16 - two_h3_real * COS__5PI_OVER_16;

	/* Store outputs */
	f[ 0 * f_stride] = std::real<float>(w0) + std::imag<float>(w0);
	f[ 1 * f_stride] = std::real<float>(w0) - std::imag<float>(w0);
	f[ 2 * f_stride] = 0.5f * two_w1_real;
	f[ 3 * f_stride] = 0.5f * two_w1_imag;
	f[ 4 * f_stride] = 0.5f * two_w2_real;
	f[ 5 * f_stride] = 0.5f * two_w2_imag;
	f[ 6 * f_stride] = 0.5f * two_w3_real;
	f[ 7 * f_stride] = 0.5f * two_w3_imag;
	f[ 8 * f_stride] = 0.5f * two_w4_real;
	f[ 9 * f_stride] = 0.5f * two_w4_imag;
	f[10 * f_stride] = 0.5f * two_w5_real;
	f[11 * f_stride] = 0.5f * two_w5_imag;
	f[12 * f_stride] = 0.5f * two_w6_real;
	f[13 * f_stride] = 0.5f * two_w6_imag;
	f[14 * f_stride] = 0.5f * two_w7_real;
	f[15 * f_stride] = 0.5f * two_w7_imag;
	f[16 * f_stride] = std::real<float>(w8);
	f[17 * f_stride] = -std::imag<float>(w8);
	f[18 * f_stride] = 0.5f * two_w9_real;
	f[19 * f_stride] = 0.5f * two_w9_imag;
	f[20 * f_stride] = 0.5f * two_w10_real;
	f[21 * f_stride] = 0.5f * two_w10_imag;
	f[22 * f_stride] = 0.5f * two_w11_real;
	f[23 * f_stride] = 0.5f * two_w11_imag;
	f[24 * f_stride] = 0.5f * two_w12_real;
	f[25 * f_stride] = 0.5f * two_w12_imag;
	f[26 * f_stride] = 0.5f * two_w13_real;
	f[27 * f_stride] = 0.5f * two_w13_imag;
	f[28 * f_stride] = 0.5f * two_w14_real;
	f[29 * f_stride] = 0.5f * two_w14_imag;
	f[30 * f_stride] = 0.5f * two_w15_real;
	f[31 * f_stride] = 0.5f * two_w15_imag;
}
