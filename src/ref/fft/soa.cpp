#include <nnpack/fft.h>
#include <ref/fft/complex-ref.h>


void nnp_fft2_soa__ref(const float* t, size_t t_stride,	float* f, size_t f_stride)
{
	/* Load inputs */
	std::complex<float> w0 = std::complex<float>(t[0 * t_stride], t[2 * t_stride]);
	std::complex<float> w1 = std::complex<float>(t[1 * t_stride], t[3 * t_stride]);

	fft2fc(&w0, &w1);

	/* Store outputs */
	f[0 * f_stride] = std::real<float>(w0);
	f[1 * f_stride] = std::real<float>(w1);
	f[2 * f_stride] = std::imag<float>(w0);
	f[3 * f_stride] = std::imag<float>(w1);
}

void nnp_fft4_soa__ref(const float* t, size_t t_stride,	float* f, size_t f_stride)
{
	/* Load inputs */
	std::complex<float> w0 = std::complex<float>(t[0 * t_stride], t[4 * t_stride]);
	std::complex<float> w1 = std::complex<float>(t[1 * t_stride], t[5 * t_stride]);
	std::complex<float> w2 = std::complex<float>(t[2 * t_stride], t[6 * t_stride]);
	std::complex<float> w3 = std::complex<float>(t[3 * t_stride], t[7 * t_stride]);

	fft4fc(&w0, &w1, &w2, &w3);

	/* Store outputs */
	f[0 * f_stride] = std::real<float>(w0);
	f[1 * f_stride] = std::real<float>(w1);
	f[2 * f_stride] = std::real<float>(w2);
	f[3 * f_stride] = std::real<float>(w3);
	f[4 * f_stride] = std::imag<float>(w0);
	f[5 * f_stride] = std::imag<float>(w1);
	f[6 * f_stride] = std::imag<float>(w2);
	f[7 * f_stride] = std::imag<float>(w3);
}

void nnp_fft8_soa__ref(const float* t, size_t t_stride,	float* f, size_t f_stride)
{
	/* Load inputs */
	std::complex<float> w0 = std::complex<float>(t[0 * t_stride], t[ 8 * t_stride]);
	std::complex<float> w1 = std::complex<float>(t[1 * t_stride], t[ 9 * t_stride]);
	std::complex<float> w2 = std::complex<float>(t[2 * t_stride], t[10 * t_stride]);
	std::complex<float> w3 = std::complex<float>(t[3 * t_stride], t[11 * t_stride]);
	std::complex<float> w4 = std::complex<float>(t[4 * t_stride], t[12 * t_stride]);
	std::complex<float> w5 = std::complex<float>(t[5 * t_stride], t[13 * t_stride]);
	std::complex<float> w6 = std::complex<float>(t[6 * t_stride], t[14 * t_stride]);
	std::complex<float> w7 = std::complex<float>(t[7 * t_stride], t[15 * t_stride]);

	fft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	/* Store outputs */
	f[ 0 * f_stride] = std::real<float>(w0);
	f[ 1 * f_stride] = std::real<float>(w1);
	f[ 2 * f_stride] = std::real<float>(w2);
	f[ 3 * f_stride] = std::real<float>(w3);
	f[ 4 * f_stride] = std::real<float>(w4);
	f[ 5 * f_stride] = std::real<float>(w5);
	f[ 6 * f_stride] = std::real<float>(w6);
	f[ 7 * f_stride] = std::real<float>(w7);
	f[ 8 * f_stride] = std::imag<float>(w0);
	f[ 9 * f_stride] = std::imag<float>(w1);
	f[10 * f_stride] = std::imag<float>(w2);
	f[11 * f_stride] = std::imag<float>(w3);
	f[12 * f_stride] = std::imag<float>(w4);
	f[13 * f_stride] = std::imag<float>(w5);
	f[14 * f_stride] = std::imag<float>(w6);
	f[15 * f_stride] = std::imag<float>(w7);
}

void nnp_fft16_soa__ref(const float* t, size_t t_stride, float* f, size_t f_stride)
{
	/* Load inputs */
	std::complex<float> w0  = std::complex<float>(t[ 0 * t_stride], t[16 * t_stride]);
	std::complex<float> w1  = std::complex<float>(t[ 1 * t_stride], t[17 * t_stride]);
	std::complex<float> w2  = std::complex<float>(t[ 2 * t_stride], t[18 * t_stride]);
	std::complex<float> w3  = std::complex<float>(t[ 3 * t_stride], t[19 * t_stride]);
	std::complex<float> w4  = std::complex<float>(t[ 4 * t_stride], t[20 * t_stride]);
	std::complex<float> w5  = std::complex<float>(t[ 5 * t_stride], t[21 * t_stride]);
	std::complex<float> w6  = std::complex<float>(t[ 6 * t_stride], t[22 * t_stride]);
	std::complex<float> w7  = std::complex<float>(t[ 7 * t_stride], t[23 * t_stride]);
	std::complex<float> w8  = std::complex<float>(t[ 8 * t_stride], t[24 * t_stride]);
	std::complex<float> w9  = std::complex<float>(t[ 9 * t_stride], t[25 * t_stride]);
	std::complex<float> w10 = std::complex<float>(t[10 * t_stride], t[26 * t_stride]);
	std::complex<float> w11 = std::complex<float>(t[11 * t_stride], t[27 * t_stride]);
	std::complex<float> w12 = std::complex<float>(t[12 * t_stride], t[28 * t_stride]);
	std::complex<float> w13 = std::complex<float>(t[13 * t_stride], t[29 * t_stride]);
	std::complex<float> w14 = std::complex<float>(t[14 * t_stride], t[30 * t_stride]);
	std::complex<float> w15 = std::complex<float>(t[15 * t_stride], t[31 * t_stride]);

	fft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	/* Store outputs */
	f[ 0 * f_stride] = std::real<float>(w0);
	f[ 1 * f_stride] = std::real<float>(w1);
	f[ 2 * f_stride] = std::real<float>(w2);
	f[ 3 * f_stride] = std::real<float>(w3);
	f[ 4 * f_stride] = std::real<float>(w4);
	f[ 5 * f_stride] = std::real<float>(w5);
	f[ 6 * f_stride] = std::real<float>(w6);
	f[ 7 * f_stride] = std::real<float>(w7);
	f[ 8 * f_stride] = std::real<float>(w8);
	f[ 9 * f_stride] = std::real<float>(w9);
	f[10 * f_stride] = std::real<float>(w10);
	f[11 * f_stride] = std::real<float>(w11);
	f[12 * f_stride] = std::real<float>(w12);
	f[13 * f_stride] = std::real<float>(w13);
	f[14 * f_stride] = std::real<float>(w14);
	f[15 * f_stride] = std::real<float>(w15);
	f[16 * f_stride] = std::imag<float>(w0);
	f[17 * f_stride] = std::imag<float>(w1);
	f[18 * f_stride] = std::imag<float>(w2);
	f[19 * f_stride] = std::imag<float>(w3);
	f[20 * f_stride] = std::imag<float>(w4);
	f[21 * f_stride] = std::imag<float>(w5);
	f[22 * f_stride] = std::imag<float>(w6);
	f[23 * f_stride] = std::imag<float>(w7);
	f[24 * f_stride] = std::imag<float>(w8);
	f[25 * f_stride] = std::imag<float>(w9);
	f[26 * f_stride] = std::imag<float>(w10);
	f[27 * f_stride] = std::imag<float>(w11);
	f[28 * f_stride] = std::imag<float>(w12);
	f[29 * f_stride] = std::imag<float>(w13);
	f[30 * f_stride] = std::imag<float>(w14);
	f[31 * f_stride] = std::imag<float>(w15);
}

void nnp_fft32_soa__ref(const float* t, size_t t_stride, float* f, size_t f_stride)
{
	/* Load inputs */
	std::complex<float> w0  = std::complex<float>(t[ 0 * t_stride], t[32 * t_stride]);
	std::complex<float> w1  = std::complex<float>(t[ 1 * t_stride], t[33 * t_stride]);
	std::complex<float> w2  = std::complex<float>(t[ 2 * t_stride], t[34 * t_stride]);
	std::complex<float> w3  = std::complex<float>(t[ 3 * t_stride], t[35 * t_stride]);
	std::complex<float> w4  = std::complex<float>(t[ 4 * t_stride], t[36 * t_stride]);
	std::complex<float> w5  = std::complex<float>(t[ 5 * t_stride], t[37 * t_stride]);
	std::complex<float> w6  = std::complex<float>(t[ 6 * t_stride], t[38 * t_stride]);
	std::complex<float> w7  = std::complex<float>(t[ 7 * t_stride], t[39 * t_stride]);
	std::complex<float> w8  = std::complex<float>(t[ 8 * t_stride], t[40 * t_stride]);
	std::complex<float> w9  = std::complex<float>(t[ 9 * t_stride], t[41 * t_stride]);
	std::complex<float> w10 = std::complex<float>(t[10 * t_stride], t[42 * t_stride]);
	std::complex<float> w11 = std::complex<float>(t[11 * t_stride], t[43 * t_stride]);
	std::complex<float> w12 = std::complex<float>(t[12 * t_stride], t[44 * t_stride]);
	std::complex<float> w13 = std::complex<float>(t[13 * t_stride], t[45 * t_stride]);
	std::complex<float> w14 = std::complex<float>(t[14 * t_stride], t[46 * t_stride]);
	std::complex<float> w15 = std::complex<float>(t[15 * t_stride], t[47 * t_stride]);
	std::complex<float> w16 = std::complex<float>(t[16 * t_stride], t[48 * t_stride]);
	std::complex<float> w17 = std::complex<float>(t[17 * t_stride], t[49 * t_stride]);
	std::complex<float> w18 = std::complex<float>(t[18 * t_stride], t[50 * t_stride]);
	std::complex<float> w19 = std::complex<float>(t[19 * t_stride], t[51 * t_stride]);
	std::complex<float> w20 = std::complex<float>(t[20 * t_stride], t[52 * t_stride]);
	std::complex<float> w21 = std::complex<float>(t[21 * t_stride], t[53 * t_stride]);
	std::complex<float> w22 = std::complex<float>(t[22 * t_stride], t[54 * t_stride]);
	std::complex<float> w23 = std::complex<float>(t[23 * t_stride], t[55 * t_stride]);
	std::complex<float> w24 = std::complex<float>(t[24 * t_stride], t[56 * t_stride]);
	std::complex<float> w25 = std::complex<float>(t[25 * t_stride], t[57 * t_stride]);
	std::complex<float> w26 = std::complex<float>(t[26 * t_stride], t[58 * t_stride]);
	std::complex<float> w27 = std::complex<float>(t[27 * t_stride], t[59 * t_stride]);
	std::complex<float> w28 = std::complex<float>(t[28 * t_stride], t[60 * t_stride]);
	std::complex<float> w29 = std::complex<float>(t[29 * t_stride], t[61 * t_stride]);
	std::complex<float> w30 = std::complex<float>(t[30 * t_stride], t[62 * t_stride]);
	std::complex<float> w31 = std::complex<float>(t[31 * t_stride], t[63 * t_stride]);

	fft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	/* Store outputs */
	f[ 0 * f_stride] = std::real<float>(w0);
	f[ 1 * f_stride] = std::real<float>(w1);
	f[ 2 * f_stride] = std::real<float>(w2);
	f[ 3 * f_stride] = std::real<float>(w3);
	f[ 4 * f_stride] = std::real<float>(w4);
	f[ 5 * f_stride] = std::real<float>(w5);
	f[ 6 * f_stride] = std::real<float>(w6);
	f[ 7 * f_stride] = std::real<float>(w7);
	f[ 8 * f_stride] = std::real<float>(w8);
	f[ 9 * f_stride] = std::real<float>(w9);
	f[10 * f_stride] = std::real<float>(w10);
	f[11 * f_stride] = std::real<float>(w11);
	f[12 * f_stride] = std::real<float>(w12);
	f[13 * f_stride] = std::real<float>(w13);
	f[14 * f_stride] = std::real<float>(w14);
	f[15 * f_stride] = std::real<float>(w15);
	f[16 * f_stride] = std::real<float>(w16);
	f[17 * f_stride] = std::real<float>(w17);
	f[18 * f_stride] = std::real<float>(w18);
	f[19 * f_stride] = std::real<float>(w19);
	f[20 * f_stride] = std::real<float>(w20);
	f[21 * f_stride] = std::real<float>(w21);
	f[22 * f_stride] = std::real<float>(w22);
	f[23 * f_stride] = std::real<float>(w23);
	f[24 * f_stride] = std::real<float>(w24);
	f[25 * f_stride] = std::real<float>(w25);
	f[26 * f_stride] = std::real<float>(w26);
	f[27 * f_stride] = std::real<float>(w27);
	f[28 * f_stride] = std::real<float>(w28);
	f[29 * f_stride] = std::real<float>(w29);
	f[30 * f_stride] = std::real<float>(w30);
	f[31 * f_stride] = std::real<float>(w31);
	f[32 * f_stride] = std::imag<float>(w0);
	f[33 * f_stride] = std::imag<float>(w1);
	f[34 * f_stride] = std::imag<float>(w2);
	f[35 * f_stride] = std::imag<float>(w3);
	f[36 * f_stride] = std::imag<float>(w4);
	f[37 * f_stride] = std::imag<float>(w5);
	f[38 * f_stride] = std::imag<float>(w6);
	f[39 * f_stride] = std::imag<float>(w7);
	f[40 * f_stride] = std::imag<float>(w8);
	f[41 * f_stride] = std::imag<float>(w9);
	f[42 * f_stride] = std::imag<float>(w10);
	f[43 * f_stride] = std::imag<float>(w11);
	f[44 * f_stride] = std::imag<float>(w12);
	f[45 * f_stride] = std::imag<float>(w13);
	f[46 * f_stride] = std::imag<float>(w14);
	f[47 * f_stride] = std::imag<float>(w15);
	f[48 * f_stride] = std::imag<float>(w16);
	f[49 * f_stride] = std::imag<float>(w17);
	f[50 * f_stride] = std::imag<float>(w18);
	f[51 * f_stride] = std::imag<float>(w19);
	f[52 * f_stride] = std::imag<float>(w20);
	f[53 * f_stride] = std::imag<float>(w21);
	f[54 * f_stride] = std::imag<float>(w22);
	f[55 * f_stride] = std::imag<float>(w23);
	f[56 * f_stride] = std::imag<float>(w24);
	f[57 * f_stride] = std::imag<float>(w25);
	f[58 * f_stride] = std::imag<float>(w26);
	f[59 * f_stride] = std::imag<float>(w27);
	f[60 * f_stride] = std::imag<float>(w28);
	f[61 * f_stride] = std::imag<float>(w29);
	f[62 * f_stride] = std::imag<float>(w30);
	f[63 * f_stride] = std::imag<float>(w31);
}

void nnp_ifft2_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride)
{
	/* Load inputs */
	std::complex<float> w0 = std::complex<float>(f[0 * f_stride], f[2 * f_stride]);
	std::complex<float> w1 = std::complex<float>(f[1 * f_stride], f[3 * f_stride]);

	ifft2fc(&w0, &w1);

	/* Store outputs */
	t[0 * t_stride] = std::real<float>(w0);
	t[1 * t_stride] = std::real<float>(w1);
	t[2 * t_stride] = std::imag<float>(w0);
	t[3 * t_stride] = std::imag<float>(w1);
}

void nnp_ifft4_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride)
{
	/* Load inputs */
	std::complex<float> w0 = std::complex<float>(f[0 * f_stride], f[4 * f_stride]);
	std::complex<float> w1 = std::complex<float>(f[1 * f_stride], f[5 * f_stride]);
	std::complex<float> w2 = std::complex<float>(f[2 * f_stride], f[6 * f_stride]);
	std::complex<float> w3 = std::complex<float>(f[3 * f_stride], f[7 * f_stride]);

	ifft4fc(&w0, &w1, &w2, &w3);

	/* Store outputs */
	t[0 * t_stride] = std::real<float>(w0);
	t[1 * t_stride] = std::real<float>(w1);
	t[2 * t_stride] = std::real<float>(w2);
	t[3 * t_stride] = std::real<float>(w3);
	t[4 * t_stride] = std::imag<float>(w0);
	t[5 * t_stride] = std::imag<float>(w1);
	t[6 * t_stride] = std::imag<float>(w2);
	t[7 * t_stride] = std::imag<float>(w3);
}

void nnp_ifft8_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride)
{
	/* Load inputs */
	std::complex<float> w0 = std::complex<float>(f[0 * f_stride], f[ 8 * f_stride]);
	std::complex<float> w1 = std::complex<float>(f[1 * f_stride], f[ 9 * f_stride]);
	std::complex<float> w2 = std::complex<float>(f[2 * f_stride], f[10 * f_stride]);
	std::complex<float> w3 = std::complex<float>(f[3 * f_stride], f[11 * f_stride]);
	std::complex<float> w4 = std::complex<float>(f[4 * f_stride], f[12 * f_stride]);
	std::complex<float> w5 = std::complex<float>(f[5 * f_stride], f[13 * f_stride]);
	std::complex<float> w6 = std::complex<float>(f[6 * f_stride], f[14 * f_stride]);
	std::complex<float> w7 = std::complex<float>(f[7 * f_stride], f[15 * f_stride]);

	ifft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	/* Store outputs */
	t[ 0 * t_stride] = std::real<float>(w0);
	t[ 1 * t_stride] = std::real<float>(w1);
	t[ 2 * t_stride] = std::real<float>(w2);
	t[ 3 * t_stride] = std::real<float>(w3);
	t[ 4 * t_stride] = std::real<float>(w4);
	t[ 5 * t_stride] = std::real<float>(w5);
	t[ 6 * t_stride] = std::real<float>(w6);
	t[ 7 * t_stride] = std::real<float>(w7);
	t[ 8 * t_stride] = std::imag<float>(w0);
	t[ 9 * t_stride] = std::imag<float>(w1);
	t[10 * t_stride] = std::imag<float>(w2);
	t[11 * t_stride] = std::imag<float>(w3);
	t[12 * t_stride] = std::imag<float>(w4);
	t[13 * t_stride] = std::imag<float>(w5);
	t[14 * t_stride] = std::imag<float>(w6);
	t[15 * t_stride] = std::imag<float>(w7);
}

void nnp_ifft16_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride)
{
	/* Load inputs */
	std::complex<float> w0  = std::complex<float>(f[ 0 * f_stride], f[16 * f_stride]);
	std::complex<float> w1  = std::complex<float>(f[ 1 * f_stride], f[17 * f_stride]);
	std::complex<float> w2  = std::complex<float>(f[ 2 * f_stride], f[18 * f_stride]);
	std::complex<float> w3  = std::complex<float>(f[ 3 * f_stride], f[19 * f_stride]);
	std::complex<float> w4  = std::complex<float>(f[ 4 * f_stride], f[20 * f_stride]);
	std::complex<float> w5  = std::complex<float>(f[ 5 * f_stride], f[21 * f_stride]);
	std::complex<float> w6  = std::complex<float>(f[ 6 * f_stride], f[22 * f_stride]);
	std::complex<float> w7  = std::complex<float>(f[ 7 * f_stride], f[23 * f_stride]);
	std::complex<float> w8  = std::complex<float>(f[ 8 * f_stride], f[24 * f_stride]);
	std::complex<float> w9  = std::complex<float>(f[ 9 * f_stride], f[25 * f_stride]);
	std::complex<float> w10 = std::complex<float>(f[10 * f_stride], f[26 * f_stride]);
	std::complex<float> w11 = std::complex<float>(f[11 * f_stride], f[27 * f_stride]);
	std::complex<float> w12 = std::complex<float>(f[12 * f_stride], f[28 * f_stride]);
	std::complex<float> w13 = std::complex<float>(f[13 * f_stride], f[29 * f_stride]);
	std::complex<float> w14 = std::complex<float>(f[14 * f_stride], f[30 * f_stride]);
	std::complex<float> w15 = std::complex<float>(f[15 * f_stride], f[31 * f_stride]);

	ifft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	/* Store outputs */
	t[ 0 * t_stride] = std::real<float>(w0);
	t[ 1 * t_stride] = std::real<float>(w1);
	t[ 2 * t_stride] = std::real<float>(w2);
	t[ 3 * t_stride] = std::real<float>(w3);
	t[ 4 * t_stride] = std::real<float>(w4);
	t[ 5 * t_stride] = std::real<float>(w5);
	t[ 6 * t_stride] = std::real<float>(w6);
	t[ 7 * t_stride] = std::real<float>(w7);
	t[ 8 * t_stride] = std::real<float>(w8);
	t[ 9 * t_stride] = std::real<float>(w9);
	t[10 * t_stride] = std::real<float>(w10);
	t[11 * t_stride] = std::real<float>(w11);
	t[12 * t_stride] = std::real<float>(w12);
	t[13 * t_stride] = std::real<float>(w13);
	t[14 * t_stride] = std::real<float>(w14);
	t[15 * t_stride] = std::real<float>(w15);
	t[16 * t_stride] = std::imag<float>(w0);
	t[17 * t_stride] = std::imag<float>(w1);
	t[18 * t_stride] = std::imag<float>(w2);
	t[19 * t_stride] = std::imag<float>(w3);
	t[20 * t_stride] = std::imag<float>(w4);
	t[21 * t_stride] = std::imag<float>(w5);
	t[22 * t_stride] = std::imag<float>(w6);
	t[23 * t_stride] = std::imag<float>(w7);
	t[24 * t_stride] = std::imag<float>(w8);
	t[25 * t_stride] = std::imag<float>(w9);
	t[26 * t_stride] = std::imag<float>(w10);
	t[27 * t_stride] = std::imag<float>(w11);
	t[28 * t_stride] = std::imag<float>(w12);
	t[29 * t_stride] = std::imag<float>(w13);
	t[30 * t_stride] = std::imag<float>(w14);
	t[31 * t_stride] = std::imag<float>(w15);
}

void nnp_ifft32_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride)
{
	/* Load inputs */
	std::complex<float> w0  = std::complex<float>(f[ 0 * f_stride], f[32 * f_stride]);
	std::complex<float> w1  = std::complex<float>(f[ 1 * f_stride], f[33 * f_stride]);
	std::complex<float> w2  = std::complex<float>(f[ 2 * f_stride], f[34 * f_stride]);
	std::complex<float> w3  = std::complex<float>(f[ 3 * f_stride], f[35 * f_stride]);
	std::complex<float> w4  = std::complex<float>(f[ 4 * f_stride], f[36 * f_stride]);
	std::complex<float> w5  = std::complex<float>(f[ 5 * f_stride], f[37 * f_stride]);
	std::complex<float> w6  = std::complex<float>(f[ 6 * f_stride], f[38 * f_stride]);
	std::complex<float> w7  = std::complex<float>(f[ 7 * f_stride], f[39 * f_stride]);
	std::complex<float> w8  = std::complex<float>(f[ 8 * f_stride], f[40 * f_stride]);
	std::complex<float> w9  = std::complex<float>(f[ 9 * f_stride], f[41 * f_stride]);
	std::complex<float> w10 = std::complex<float>(f[10 * f_stride], f[42 * f_stride]);
	std::complex<float> w11 = std::complex<float>(f[11 * f_stride], f[43 * f_stride]);
	std::complex<float> w12 = std::complex<float>(f[12 * f_stride], f[44 * f_stride]);
	std::complex<float> w13 = std::complex<float>(f[13 * f_stride], f[45 * f_stride]);
	std::complex<float> w14 = std::complex<float>(f[14 * f_stride], f[46 * f_stride]);
	std::complex<float> w15 = std::complex<float>(f[15 * f_stride], f[47 * f_stride]);
	std::complex<float> w16 = std::complex<float>(f[16 * f_stride], f[48 * f_stride]);
	std::complex<float> w17 = std::complex<float>(f[17 * f_stride], f[49 * f_stride]);
	std::complex<float> w18 = std::complex<float>(f[18 * f_stride], f[50 * f_stride]);
	std::complex<float> w19 = std::complex<float>(f[19 * f_stride], f[51 * f_stride]);
	std::complex<float> w20 = std::complex<float>(f[20 * f_stride], f[52 * f_stride]);
	std::complex<float> w21 = std::complex<float>(f[21 * f_stride], f[53 * f_stride]);
	std::complex<float> w22 = std::complex<float>(f[22 * f_stride], f[54 * f_stride]);
	std::complex<float> w23 = std::complex<float>(f[23 * f_stride], f[55 * f_stride]);
	std::complex<float> w24 = std::complex<float>(f[24 * f_stride], f[56 * f_stride]);
	std::complex<float> w25 = std::complex<float>(f[25 * f_stride], f[57 * f_stride]);
	std::complex<float> w26 = std::complex<float>(f[26 * f_stride], f[58 * f_stride]);
	std::complex<float> w27 = std::complex<float>(f[27 * f_stride], f[59 * f_stride]);
	std::complex<float> w28 = std::complex<float>(f[28 * f_stride], f[60 * f_stride]);
	std::complex<float> w29 = std::complex<float>(f[29 * f_stride], f[61 * f_stride]);
	std::complex<float> w30 = std::complex<float>(f[30 * f_stride], f[62 * f_stride]);
	std::complex<float> w31 = std::complex<float>(f[31 * f_stride], f[63 * f_stride]);

	ifft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	/* Store outputs */
	t[ 0 * t_stride] = std::real<float>(w0);
	t[ 1 * t_stride] = std::real<float>(w1);
	t[ 2 * t_stride] = std::real<float>(w2);
	t[ 3 * t_stride] = std::real<float>(w3);
	t[ 4 * t_stride] = std::real<float>(w4);
	t[ 5 * t_stride] = std::real<float>(w5);
	t[ 6 * t_stride] = std::real<float>(w6);
	t[ 7 * t_stride] = std::real<float>(w7);
	t[ 8 * t_stride] = std::real<float>(w8);
	t[ 9 * t_stride] = std::real<float>(w9);
	t[10 * t_stride] = std::real<float>(w10);
	t[11 * t_stride] = std::real<float>(w11);
	t[12 * t_stride] = std::real<float>(w12);
	t[13 * t_stride] = std::real<float>(w13);
	t[14 * t_stride] = std::real<float>(w14);
	t[15 * t_stride] = std::real<float>(w15);
	t[16 * t_stride] = std::real<float>(w16);
	t[17 * t_stride] = std::real<float>(w17);
	t[18 * t_stride] = std::real<float>(w18);
	t[19 * t_stride] = std::real<float>(w19);
	t[20 * t_stride] = std::real<float>(w20);
	t[21 * t_stride] = std::real<float>(w21);
	t[22 * t_stride] = std::real<float>(w22);
	t[23 * t_stride] = std::real<float>(w23);
	t[24 * t_stride] = std::real<float>(w24);
	t[25 * t_stride] = std::real<float>(w25);
	t[26 * t_stride] = std::real<float>(w26);
	t[27 * t_stride] = std::real<float>(w27);
	t[28 * t_stride] = std::real<float>(w28);
	t[29 * t_stride] = std::real<float>(w29);
	t[30 * t_stride] = std::real<float>(w30);
	t[31 * t_stride] = std::real<float>(w31);
	t[32 * t_stride] = std::imag<float>(w0);
	t[33 * t_stride] = std::imag<float>(w1);
	t[34 * t_stride] = std::imag<float>(w2);
	t[35 * t_stride] = std::imag<float>(w3);
	t[36 * t_stride] = std::imag<float>(w4);
	t[37 * t_stride] = std::imag<float>(w5);
	t[38 * t_stride] = std::imag<float>(w6);
	t[39 * t_stride] = std::imag<float>(w7);
	t[40 * t_stride] = std::imag<float>(w8);
	t[41 * t_stride] = std::imag<float>(w9);
	t[42 * t_stride] = std::imag<float>(w10);
	t[43 * t_stride] = std::imag<float>(w11);
	t[44 * t_stride] = std::imag<float>(w12);
	t[45 * t_stride] = std::imag<float>(w13);
	t[46 * t_stride] = std::imag<float>(w14);
	t[47 * t_stride] = std::imag<float>(w15);
	t[48 * t_stride] = std::imag<float>(w16);
	t[49 * t_stride] = std::imag<float>(w17);
	t[50 * t_stride] = std::imag<float>(w18);
	t[51 * t_stride] = std::imag<float>(w19);
	t[52 * t_stride] = std::imag<float>(w20);
	t[53 * t_stride] = std::imag<float>(w21);
	t[54 * t_stride] = std::imag<float>(w22);
	t[55 * t_stride] = std::imag<float>(w23);
	t[56 * t_stride] = std::imag<float>(w24);
	t[57 * t_stride] = std::imag<float>(w25);
	t[58 * t_stride] = std::imag<float>(w26);
	t[59 * t_stride] = std::imag<float>(w27);
	t[60 * t_stride] = std::imag<float>(w28);
	t[61 * t_stride] = std::imag<float>(w29);
	t[62 * t_stride] = std::imag<float>(w30);
	t[63 * t_stride] = std::imag<float>(w31);
}
