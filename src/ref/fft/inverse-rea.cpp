#include <nnpack/fft.h>
#include <ref/fft/complex-ref.h>

/**
 * From
 *
 * g[ n] =  0.5  (w[ n] + conj w[-n])
 * g[-n] =  0.5  (w[-n] + conj w[ n]) = conj g[n]
 * h[ n] = -0.5i (w[ n] - conj w[-n])
 * h[-n] = -0.5i (w[-n] - conj w[ n]) = conj h[n]
 *
 * ...and...
 *
 * w[ n] = g[ n] + h[ n] * conj CEXP_nPI_OVER_N
 * w[-n] = g[-n] + h[-n] * conj CEXP_nPI_OVER_N
 *
 * ...we have:
 *   w[-n] =      g[-n] +      h[-n] * conj CEXP_nPI_OVER_N
 *         = conj g[ n] + conj h[ n] * conj CEXP_nPI_OVER_N
 *
 * Therefore
 *   conj w[-n] = g[n] + h[n] * CEXP_nPI_OVER_N
 *              = g[n] - h[n] * conj CEXP_nPI_OVER_N
 *        w[ n] = g[n] + h[n] * conj CEXP_nPI_OVER_N
 *
 * From there:
 *   g[n]                        = 0.5 (w[n] + conj w[-n])
 *   h[n] * conj CEXP_nPI_OVER_N = 0.5 (w[n] - conj w[-n])
 *   h[n]                        = 0.5 (w[n] - conj w[-n]) * CEXP_nPI_OVER_N
 */

void nnp_ifft8_real__ref(const float* f, size_t f_stride, float* t, size_t t_stride)
{
	/* Load inputs and scale */
	const float W0_real = 0.5f * f[0 * f_stride];
	const float W4_real = 0.5f * f[1 * f_stride];
	const float W1_real = 0.5f * f[2 * f_stride];
	const float W1_imag = 0.5f * f[3 * f_stride];
	const float W2_real =        f[4 * f_stride];
	const float W2_imag =        f[5 * f_stride];
	const float W3_real = 0.5f * f[6 * f_stride];
	const float W3_imag = 0.5f * f[7 * f_stride];

	std::complex<float> w0 = std::complex<float>(W0_real + W4_real, W0_real - W4_real);
	std::complex<float> w2 = std::complex<float>(W2_real, -W2_imag);

	const float G1_real = W1_real + W3_real;
	const float G1_imag = W1_imag - W3_imag;

	const float H1_real = W1_real - W3_real;
	const float H1_imag = W1_imag + W3_imag;

	const float H1_plus  = H1_real + H1_imag;
	const float H1_minus = H1_real - H1_imag;

	const float w1_real =  G1_real - SQRT2_OVER_2 * H1_plus;
	const float w1_imag =  G1_imag + SQRT2_OVER_2 * H1_minus;
	const float w3_real =  G1_real + SQRT2_OVER_2 * H1_plus;
	const float w3_imag = -G1_imag + SQRT2_OVER_2 * H1_minus;

	std::complex<float> w1 = std::complex<float>(w1_real, w1_imag);
	std::complex<float> w3 = std::complex<float>(w3_real, w3_imag);

	ifft4fc(&w0, &w1, &w2, &w3);

	/* Store outputs */
	t[0 * t_stride] = std::real<float>(w0);
	t[1 * t_stride] = std::imag<float>(w0);
	t[2 * t_stride] = std::real<float>(w1);
	t[3 * t_stride] = std::imag<float>(w1);
	t[4 * t_stride] = std::real<float>(w2);
	t[5 * t_stride] = std::imag<float>(w2);
	t[6 * t_stride] = std::real<float>(w3);
	t[7 * t_stride] = std::imag<float>(w3);
}

void nnp_ifft16_real__ref(const float* f, size_t f_stride, float* t, size_t t_stride)
{
	/* Load inputs and scale */
	const float W0_real = 0.5f * f[ 0 * f_stride];
	const float W8_real = 0.5f * f[ 1 * f_stride];
	const float W1_real = 0.5f * f[ 2 * f_stride];
	const float W1_imag = 0.5f * f[ 3 * f_stride];
	const float W2_real = 0.5f * f[ 4 * f_stride];
	const float W2_imag = 0.5f * f[ 5 * f_stride];
	const float W3_real = 0.5f * f[ 6 * f_stride];
	const float W3_imag = 0.5f * f[ 7 * f_stride];
	const float W4_real =        f[ 8 * f_stride];
	const float W4_imag =        f[ 9 * f_stride];
	const float W5_real = 0.5f * f[10 * f_stride];
	const float W5_imag = 0.5f * f[11 * f_stride];
	const float W6_real = 0.5f * f[12 * f_stride];
	const float W6_imag = 0.5f * f[13 * f_stride];
	const float W7_real = 0.5f * f[14 * f_stride];
	const float W7_imag = 0.5f * f[15 * f_stride];

	std::complex<float> w0 = std::complex<float>(W0_real + W8_real, W0_real - W8_real);
	std::complex<float> w4 = std::complex<float>(W4_real, -W4_imag);

	const float G2_real = W2_real + W6_real;
	const float G2_imag = W2_imag - W6_imag;

	const float H2_real = W2_real - W6_real;
	const float H2_imag = W2_imag + W6_imag;

	const float H2_plus  = H2_real + H2_imag;
	const float H2_minus = H2_real - H2_imag;

	const float w2_real =  G2_real - SQRT2_OVER_2 * H2_plus;
	const float w2_imag =  G2_imag + SQRT2_OVER_2 * H2_minus;
	const float w6_real =  G2_real + SQRT2_OVER_2 * H2_plus;
	const float w6_imag = -G2_imag + SQRT2_OVER_2 * H2_minus;

	const float G1_real = W1_real + W7_real;
	const float G1_imag = W1_imag - W7_imag;
	const float G3_real = W3_real + W5_real;
	const float G3_imag = W3_imag - W5_imag;

	const float H1_real = W1_real - W7_real;
	const float H1_imag = W1_imag + W7_imag;
	const float H3_real = W3_real - W5_real;
	const float H3_imag = W3_imag + W5_imag;

	const float w1_real =  G1_real - H1_imag * COS_1PI_OVER_8 - H1_real * COS_3PI_OVER_8;
	const float w1_imag =  G1_imag + H1_real * COS_1PI_OVER_8 - H1_imag * COS_3PI_OVER_8;
	const float w7_real =  G1_real + H1_imag * COS_1PI_OVER_8 + H1_real * COS_3PI_OVER_8;
	const float w7_imag = -G1_imag + H1_real * COS_1PI_OVER_8 - H1_imag * COS_3PI_OVER_8;

	const float w3_real =  G3_real - H3_imag * COS_3PI_OVER_8 - H3_real * COS_1PI_OVER_8;
	const float w3_imag =  G3_imag + H3_real * COS_3PI_OVER_8 - H3_imag * COS_1PI_OVER_8;
	const float w5_real =  G3_real + H3_imag * COS_3PI_OVER_8 + H3_real * COS_1PI_OVER_8;
	const float w5_imag = -G3_imag + H3_real * COS_3PI_OVER_8 - H3_imag * COS_1PI_OVER_8;

	std::complex<float> w1 = std::complex<float>(w1_real, w1_imag);
	std::complex<float> w7 = std::complex<float>(w7_real, w7_imag);
	std::complex<float> w2 = std::complex<float>(w2_real, w2_imag);
	std::complex<float> w6 = std::complex<float>(w6_real, w6_imag);
	std::complex<float> w3 = std::complex<float>(w3_real, w3_imag);
	std::complex<float> w5 = std::complex<float>(w5_real, w5_imag);

	ifft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	/* Store outputs */
	t[ 0 * t_stride] = std::real<float>(w0);
	t[ 1 * t_stride] = std::imag<float>(w0);
	t[ 2 * t_stride] = std::real<float>(w1);
	t[ 3 * t_stride] = std::imag<float>(w1);
	t[ 4 * t_stride] = std::real<float>(w2);
	t[ 5 * t_stride] = std::imag<float>(w2);
	t[ 6 * t_stride] = std::real<float>(w3);
	t[ 7 * t_stride] = std::imag<float>(w3);
	t[ 8 * t_stride] = std::real<float>(w4);
	t[ 9 * t_stride] = std::imag<float>(w4);
	t[10 * t_stride] = std::real<float>(w5);
	t[11 * t_stride] = std::imag<float>(w5);
	t[12 * t_stride] = std::real<float>(w6);
	t[13 * t_stride] = std::imag<float>(w6);
	t[14 * t_stride] = std::real<float>(w7);
	t[15 * t_stride] = std::imag<float>(w7);
}

void nnp_ifft32_real__ref(const float* f, size_t f_stride, float* t, size_t t_stride)
{
	/* Load inputs and scale */
	const float  W0_real = 0.5f * f[ 0 * f_stride];
	const float W16_real = 0.5f * f[ 1 * f_stride];
	const float  W1_real = 0.5f * f[ 2 * f_stride];
	const float  W1_imag = 0.5f * f[ 3 * f_stride];
	const float  W2_real = 0.5f * f[ 4 * f_stride];
	const float  W2_imag = 0.5f * f[ 5 * f_stride];
	const float  W3_real = 0.5f * f[ 6 * f_stride];
	const float  W3_imag = 0.5f * f[ 7 * f_stride];
	const float  W4_real = 0.5f * f[ 8 * f_stride];
	const float  W4_imag = 0.5f * f[ 9 * f_stride];
	const float  W5_real = 0.5f * f[10 * f_stride];
	const float  W5_imag = 0.5f * f[11 * f_stride];
	const float  W6_real = 0.5f * f[12 * f_stride];
	const float  W6_imag = 0.5f * f[13 * f_stride];
	const float  W7_real = 0.5f * f[14 * f_stride];
	const float  W7_imag = 0.5f * f[15 * f_stride];
	const float  W8_real =        f[16 * f_stride];
	const float  W8_imag =        f[17 * f_stride];
	const float  W9_real = 0.5f * f[18 * f_stride];
	const float  W9_imag = 0.5f * f[19 * f_stride];
	const float W10_real = 0.5f * f[20 * f_stride];
	const float W10_imag = 0.5f * f[21 * f_stride];
	const float W11_real = 0.5f * f[22 * f_stride];
	const float W11_imag = 0.5f * f[23 * f_stride];
	const float W12_real = 0.5f * f[24 * f_stride];
	const float W12_imag = 0.5f * f[25 * f_stride];
	const float W13_real = 0.5f * f[26 * f_stride];
	const float W13_imag = 0.5f * f[27 * f_stride];
	const float W14_real = 0.5f * f[28 * f_stride];
	const float W14_imag = 0.5f * f[29 * f_stride];
	const float W15_real = 0.5f * f[30 * f_stride];
	const float W15_imag = 0.5f * f[31 * f_stride];

	std::complex<float> w0 = std::complex<float>(W0_real + W16_real, W0_real - W16_real);
	std::complex<float> w8 = std::complex<float>(W8_real, -W8_imag);

	const float G4_real = W4_real + W12_real;
	const float G4_imag = W4_imag - W12_imag;

	const float H4_real = W4_real - W12_real;
	const float H4_imag = W4_imag + W12_imag;

	const float H4_plus  = H4_real + H4_imag;
	const float H4_minus = H4_real - H4_imag;

	const float  w4_real =  G4_real - SQRT2_OVER_2 * H4_plus;
	const float  w4_imag =  G4_imag + SQRT2_OVER_2 * H4_minus;
	const float w12_real =  G4_real + SQRT2_OVER_2 * H4_plus;
	const float w12_imag = -G4_imag + SQRT2_OVER_2 * H4_minus;
	std::complex<float> w4  = std::complex<float>( w4_real,  w4_imag);
	std::complex<float> w12 = std::complex<float>(w12_real, w12_imag);

	const float G1_real = W1_real + W15_real;
	const float G1_imag = W1_imag - W15_imag;
	const float G7_real = W7_real + W9_real;
	const float G7_imag = W7_imag - W9_imag;

	const float H1_real = W1_real - W15_real;
	const float H1_imag = W1_imag + W15_imag;
	const float H7_real = W7_real - W9_real;
	const float H7_imag = W7_imag + W9_imag;

	const float  w1_real =  G1_real - H1_imag * COS__1PI_OVER_16 - H1_real * COS__7PI_OVER_16;
	const float  w1_imag =  G1_imag + H1_real * COS__1PI_OVER_16 - H1_imag * COS__7PI_OVER_16;
	const float w15_real =  G1_real + H1_imag * COS__1PI_OVER_16 + H1_real * COS__7PI_OVER_16;
	const float w15_imag = -G1_imag + H1_real * COS__1PI_OVER_16 - H1_imag * COS__7PI_OVER_16;
	std::complex<float> w1  = std::complex<float>( w1_real,  w1_imag);
	std::complex<float> w15 = std::complex<float>(w15_real, w15_imag);

	const float w7_real =  G7_real - H7_imag * COS__7PI_OVER_16 - H7_real * COS__1PI_OVER_16;
	const float w7_imag =  G7_imag + H7_real * COS__7PI_OVER_16 - H7_imag * COS__1PI_OVER_16;
	const float w9_real =  G7_real + H7_imag * COS__7PI_OVER_16 + H7_real * COS__1PI_OVER_16;
	const float w9_imag = -G7_imag + H7_real * COS__7PI_OVER_16 - H7_imag * COS__1PI_OVER_16;
	std::complex<float> w7 = std::complex<float>(w7_real, w7_imag);
	std::complex<float> w9 = std::complex<float>(w9_real, w9_imag);

	const float G2_real = W2_real + W14_real;
	const float G2_imag = W2_imag - W14_imag;
	const float G6_real = W6_real + W10_real;
	const float G6_imag = W6_imag - W10_imag;

	const float H2_real = W2_real - W14_real;
	const float H2_imag = W2_imag + W14_imag;
	const float H6_real = W6_real - W10_real;
	const float H6_imag = W6_imag + W10_imag;

	const float  w2_real =  G2_real - H2_imag * COS__2PI_OVER_16 - H2_real * COS__6PI_OVER_16;
	const float  w2_imag =  G2_imag + H2_real * COS__2PI_OVER_16 - H2_imag * COS__6PI_OVER_16;
	const float w14_real =  G2_real + H2_imag * COS__2PI_OVER_16 + H2_real * COS__6PI_OVER_16;
	const float w14_imag = -G2_imag + H2_real * COS__2PI_OVER_16 - H2_imag * COS__6PI_OVER_16;
	std::complex<float> w2  = std::complex<float>( w2_real,  w2_imag);
	std::complex<float> w14 = std::complex<float>(w14_real, w14_imag);

	const float  w6_real =  G6_real - H6_imag * COS__6PI_OVER_16 - H6_real * COS__2PI_OVER_16;
	const float  w6_imag =  G6_imag + H6_real * COS__6PI_OVER_16 - H6_imag * COS__2PI_OVER_16;
	const float w10_real =  G6_real + H6_imag * COS__6PI_OVER_16 + H6_real * COS__2PI_OVER_16;
	const float w10_imag = -G6_imag + H6_real * COS__6PI_OVER_16 - H6_imag * COS__2PI_OVER_16;
	std::complex<float> w6  = std::complex<float>( w6_real,  w6_imag);
	std::complex<float> w10 = std::complex<float>(w10_real, w10_imag);

	const float G3_real = W3_real + W13_real;
	const float G3_imag = W3_imag - W13_imag;
	const float G5_real = W5_real + W11_real;
	const float G5_imag = W5_imag - W11_imag;

	const float H3_real = W3_real - W13_real;
	const float H3_imag = W3_imag + W13_imag;
	const float H5_real = W5_real - W11_real;
	const float H5_imag = W5_imag + W11_imag;

	const float  w3_real =  G3_real - H3_imag * COS__3PI_OVER_16 - H3_real * COS__5PI_OVER_16;
	const float  w3_imag =  G3_imag + H3_real * COS__3PI_OVER_16 - H3_imag * COS__5PI_OVER_16;
	const float w13_real =  G3_real + H3_imag * COS__3PI_OVER_16 + H3_real * COS__5PI_OVER_16;
	const float w13_imag = -G3_imag + H3_real * COS__3PI_OVER_16 - H3_imag * COS__5PI_OVER_16;
	std::complex<float> w3  = std::complex<float>( w3_real,  w3_imag);
	std::complex<float> w13 = std::complex<float>(w13_real, w13_imag);

	const float  w5_real =  G5_real - H5_imag * COS__5PI_OVER_16 - H5_real * COS__3PI_OVER_16;
	const float  w5_imag =  G5_imag + H5_real * COS__5PI_OVER_16 - H5_imag * COS__3PI_OVER_16;
	const float w11_real =  G5_real + H5_imag * COS__5PI_OVER_16 + H5_real * COS__3PI_OVER_16;
	const float w11_imag = -G5_imag + H5_real * COS__5PI_OVER_16 - H5_imag * COS__3PI_OVER_16;
	std::complex<float> w5  = std::complex<float>( w5_real,  w5_imag);
	std::complex<float> w11 = std::complex<float>(w11_real, w11_imag);

	ifft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	/* Store outputs */
	t[ 0 * t_stride] = std::real<float>(w0);
	t[ 1 * t_stride] = std::imag<float>(w0);
	t[ 2 * t_stride] = std::real<float>(w1);
	t[ 3 * t_stride] = std::imag<float>(w1);
	t[ 4 * t_stride] = std::real<float>(w2);
	t[ 5 * t_stride] = std::imag<float>(w2);
	t[ 6 * t_stride] = std::real<float>(w3);
	t[ 7 * t_stride] = std::imag<float>(w3);
	t[ 8 * t_stride] = std::real<float>(w4);
	t[ 9 * t_stride] = std::imag<float>(w4);
	t[10 * t_stride] = std::real<float>(w5);
	t[11 * t_stride] = std::imag<float>(w5);
	t[12 * t_stride] = std::real<float>(w6);
	t[13 * t_stride] = std::imag<float>(w6);
	t[14 * t_stride] = std::real<float>(w7);
	t[15 * t_stride] = std::imag<float>(w7);
	t[16 * t_stride] = std::real<float>(w8);
	t[17 * t_stride] = std::imag<float>(w8);
	t[18 * t_stride] = std::real<float>(w9);
	t[19 * t_stride] = std::imag<float>(w9);
	t[20 * t_stride] = std::real<float>(w10);
	t[21 * t_stride] = std::imag<float>(w10);
	t[22 * t_stride] = std::real<float>(w11);
	t[23 * t_stride] = std::imag<float>(w11);
	t[24 * t_stride] = std::real<float>(w12);
	t[25 * t_stride] = std::imag<float>(w12);
	t[26 * t_stride] = std::real<float>(w13);
	t[27 * t_stride] = std::imag<float>(w13);
	t[28 * t_stride] = std::real<float>(w14);
	t[29 * t_stride] = std::imag<float>(w14);
	t[30 * t_stride] = std::real<float>(w15);
	t[31 * t_stride] = std::imag<float>(w15);
}
