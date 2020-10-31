#include <nnpack/fft.h>
#include <ref/fft/complex-ref.h>


void nnp_fft8_dualreal__ref(const float* t, float* f) 
{
	std::complex<float> w0 = std::complex<float>(t[0], t[ 8]);
	std::complex<float> w1 = std::complex<float>(t[1], t[ 9]);
	std::complex<float> w2 = std::complex<float>(t[2], t[10]);
	std::complex<float> w3 = std::complex<float>(t[3], t[11]);
	std::complex<float> w4 = std::complex<float>(t[4], t[12]);
	std::complex<float> w5 = std::complex<float>(t[5], t[13]);
	std::complex<float> w6 = std::complex<float>(t[6], t[14]);
	std::complex<float> w7 = std::complex<float>(t[7], t[15]);

	fft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	const float x0 = std::real<float>(w0);
	const float h0 = std::imag<float>(w0);
	

	const std::complex<float>  x1 =  0.5f  * (w1 + std::conj(w7));
	const std::complex<float> h1 = std::complex<float>(0.0f, -0.5f) * (w1 - std::conj(w7));
	const std::complex<float> x2 =  0.5f  * (w2 + std::conj(w6));
	const std::complex<float> h2 = std::complex<float>(0.0f, -0.5f) * (w2 - std::conj(w6));
	const std::complex<float> x3 =  0.5f  * (w3 + std::conj(w5));
	const std::complex<float> h3 = std::complex<float>(0.0f, -0.5f) * (w3 - std::conj(w5));

	const float x4 = std::real<float>(w4);
	const float h4 = std::imag<float>(w4);

	f[0] = x0;
	f[1] = h0;
	f[2] = std::real<float>(x1);
	f[3] = std::real<float>(h1);
	f[4] = std::real<float>(x2);
	f[5] = std::real<float>(h2);
	f[6] = std::real<float>(x3);
	f[7] = std::real<float>(h3);

	f[ 8] = x4;
	f[ 9] = h4;
	f[10] = std::imag<float>(x1);
	f[11] = std::imag<float>(h1);
	f[12] = std::imag<float>(x2);
	f[13] = std::imag<float>(h2);
	f[14] = std::imag<float>(x3);
	f[15] = std::imag<float>(h3);
}

void nnp_fft16_dualreal__ref(const float* t, float* f)
{
	std::complex<float> w0  = std::complex<float>(t[ 0], t[16]);
	std::complex<float> w1  = std::complex<float>(t[ 1], t[17]);
	std::complex<float> w2  = std::complex<float>(t[ 2], t[18]);
	std::complex<float> w3  = std::complex<float>(t[ 3], t[19]);
	std::complex<float> w4  = std::complex<float>(t[ 4], t[20]);
	std::complex<float> w5  = std::complex<float>(t[ 5], t[21]);
	std::complex<float> w6  = std::complex<float>(t[ 6], t[22]);
	std::complex<float> w7  = std::complex<float>(t[ 7], t[23]);
	std::complex<float> w8  = std::complex<float>(t[ 8], t[24]);
	std::complex<float> w9  = std::complex<float>(t[ 9], t[25]);
	std::complex<float> w10 = std::complex<float>(t[10], t[26]);
	std::complex<float> w11 = std::complex<float>(t[11], t[27]);
	std::complex<float> w12 = std::complex<float>(t[12], t[28]);
	std::complex<float> w13 = std::complex<float>(t[13], t[29]);
	std::complex<float> w14 = std::complex<float>(t[14], t[30]);
	std::complex<float> w15 = std::complex<float>(t[15], t[31]);

	fft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	const float x0 = std::real<float>(w0);
	const float h0 = std::imag<float>(w0);

	const std::complex<float> x1 =  0.5f  * (w1 + std::conj(w15));
	const std::complex<float> h1 = std::complex<float>(0.0f, -0.5f) * (w1 - std::conj(w15));
	const std::complex<float> x2 =  0.5f  * (w2 + std::conj(w14));
	const std::complex<float> h2 = std::complex<float>(0.0f, -0.5f) * (w2 - std::conj(w14));
	const std::complex<float> x3 =  0.5f  * (w3 + std::conj(w13));
	const std::complex<float> h3 = std::complex<float>(0.0f, -0.5f) * (w3 - std::conj(w13));
	const std::complex<float> x4 =  0.5f  * (w4 + std::conj(w12));
	const std::complex<float> h4 = std::complex<float>(0.0f, -0.5f) * (w4 - std::conj(w12));
	const std::complex<float> x5 =  0.5f  * (w5 + std::conj(w11));
	const std::complex<float> h5 = std::complex<float>(0.0f, -0.5f) * (w5 - std::conj(w11));
	const std::complex<float> x6 =  0.5f  * (w6 + std::conj(w10));
	const std::complex<float> h6 = std::complex<float>(0.0f, -0.5f) * (w6 - std::conj(w10));
	const std::complex<float> x7 =  0.5f  * (w7 + std::conj(w9));
	const std::complex<float> h7 = std::complex<float>(0.0f, -0.5f) * (w7 - std::conj(w9));

	const float x8 = std::real<float>(w8);
	const float h8 = std::imag<float>(w8);

	f[ 0] = x0;
	f[ 1] = h0;
	f[ 2] = std::real<float>(x1);
	f[ 3] = std::real<float>(h1);
	f[ 4] = std::real<float>(x2);
	f[ 5] = std::real<float>(h2);
	f[ 6] = std::real<float>(x3);
	f[ 7] = std::real<float>(h3);
	f[ 8] = std::real<float>(x4);
	f[ 9] = std::real<float>(h4);
	f[10] = std::real<float>(x5);
	f[11] = std::real<float>(h5);
	f[12] = std::real<float>(x6);
	f[13] = std::real<float>(h6);
	f[14] = std::real<float>(x7);
	f[15] = std::real<float>(h7);

	f[16] = x8;
	f[17] = h8;
	f[18] = std::imag<float>(x1);
	f[19] = std::imag<float>(h1);
	f[20] = std::imag<float>(x2);
	f[21] = std::imag<float>(h2);
	f[22] = std::imag<float>(x3);
	f[23] = std::imag<float>(h3);
	f[24] = std::imag<float>(x4);
	f[25] = std::imag<float>(h4);
	f[26] = std::imag<float>(x5);
	f[27] = std::imag<float>(h5);
	f[28] = std::imag<float>(x6);
	f[29] = std::imag<float>(h6);
	f[30] = std::imag<float>(x7);
	f[31] = std::imag<float>(h7);
}

void nnp_fft32_dualreal__ref(const float* t, float* f) 
{
	std::complex<float> w0  = std::complex<float>(t[ 0], t[32]);
	std::complex<float> w1  = std::complex<float>(t[ 1], t[33]);
	std::complex<float> w2  = std::complex<float>(t[ 2], t[34]);
	std::complex<float> w3  = std::complex<float>(t[ 3], t[35]);
	std::complex<float> w4  = std::complex<float>(t[ 4], t[36]);
	std::complex<float> w5  = std::complex<float>(t[ 5], t[37]);
	std::complex<float> w6  = std::complex<float>(t[ 6], t[38]);
	std::complex<float> w7  = std::complex<float>(t[ 7], t[39]);
	std::complex<float> w8  = std::complex<float>(t[ 8], t[40]);
	std::complex<float> w9  = std::complex<float>(t[ 9], t[41]);
	std::complex<float> w10 = std::complex<float>(t[10], t[42]);
	std::complex<float> w11 = std::complex<float>(t[11], t[43]);
	std::complex<float> w12 = std::complex<float>(t[12], t[44]);
	std::complex<float> w13 = std::complex<float>(t[13], t[45]);
	std::complex<float> w14 = std::complex<float>(t[14], t[46]);
	std::complex<float> w15 = std::complex<float>(t[15], t[47]);
	std::complex<float> w16 = std::complex<float>(t[16], t[48]);
	std::complex<float> w17 = std::complex<float>(t[17], t[49]);
	std::complex<float> w18 = std::complex<float>(t[18], t[50]);
	std::complex<float> w19 = std::complex<float>(t[19], t[51]);
	std::complex<float> w20 = std::complex<float>(t[20], t[52]);
	std::complex<float> w21 = std::complex<float>(t[21], t[53]);
	std::complex<float> w22 = std::complex<float>(t[22], t[54]);
	std::complex<float> w23 = std::complex<float>(t[23], t[55]);
	std::complex<float> w24 = std::complex<float>(t[24], t[56]);
	std::complex<float> w25 = std::complex<float>(t[25], t[57]);
	std::complex<float> w26 = std::complex<float>(t[26], t[58]);
	std::complex<float> w27 = std::complex<float>(t[27], t[59]);
	std::complex<float> w28 = std::complex<float>(t[28], t[60]);
	std::complex<float> w29 = std::complex<float>(t[29], t[61]);
	std::complex<float> w30 = std::complex<float>(t[30], t[62]);
	std::complex<float> w31 = std::complex<float>(t[31], t[63]);

	fft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	const float x0 = std::real<float>(w0);
	const float h0 = std::imag<float>(w0);

	const std::complex<float> x1  =  0.5f  * (w1  + std::conj(w31));
	const std::complex<float> h1  = std::complex<float>(0.0f, -0.5f) * (w1  - std::conj(w31));
	const std::complex<float> x2  =  0.5f  * (w2  + std::conj(w30));
	const std::complex<float> h2  = std::complex<float>(0.0f, -0.5f) * (w2  - std::conj(w30));
	const std::complex<float> x3  =  0.5f  * (w3  + std::conj(w29));
	const std::complex<float> h3  = std::complex<float>(0.0f, -0.5f) * (w3  - std::conj(w29));
	const std::complex<float> x4  =  0.5f  * (w4  + std::conj(w28));
	const std::complex<float> h4  = std::complex<float>(0.0f, -0.5f) * (w4  - std::conj(w28));
	const std::complex<float> x5  =  0.5f  * (w5  + std::conj(w27));
	const std::complex<float> h5  = std::complex<float>(0.0f, -0.5f) * (w5  - std::conj(w27));
	const std::complex<float> x6  =  0.5f  * (w6  + std::conj(w26));
	const std::complex<float> h6  = std::complex<float>(0.0f, -0.5f) * (w6  - std::conj(w26));
	const std::complex<float> x7  =  0.5f  * (w7  + std::conj(w25));
	const std::complex<float> h7  = std::complex<float>(0.0f, -0.5f) * (w7  - std::conj(w25));
	const std::complex<float> x8  =  0.5f  * (w8  + std::conj(w24));
	const std::complex<float> h8  = std::complex<float>(0.0f, -0.5f) * (w8  - std::conj(w24));
	const std::complex<float> x9  =  0.5f  * (w9  + std::conj(w23));
	const std::complex<float> h9  = std::complex<float>(0.0f, -0.5f) * (w9  - std::conj(w23));
	const std::complex<float> x10 =  0.5f  * (w10 + std::conj(w22));
	const std::complex<float> h10 = std::complex<float>(0.0f, -0.5f) * (w10 - std::conj(w22));
	const std::complex<float> x11 =  0.5f  * (w11 + std::conj(w21));
	const std::complex<float> h11 = std::complex<float>(0.0f, -0.5f) * (w11 - std::conj(w21));
	const std::complex<float> x12 =  0.5f  * (w12 + std::conj(w20));
	const std::complex<float> h12 = std::complex<float>(0.0f, -0.5f) * (w12 - std::conj(w20));
	const std::complex<float> x13 =  0.5f  * (w13 + std::conj(w19));
	const std::complex<float> h13 = std::complex<float>(0.0f, -0.5f) * (w13 - std::conj(w19));
	const std::complex<float> x14 =  0.5f  * (w14 + std::conj(w18));
	const std::complex<float> h14 = std::complex<float>(0.0f, -0.5f) * (w14 - std::conj(w18));
	const std::complex<float> x15 =  0.5f  * (w15 + std::conj(w17));
	const std::complex<float> h15 = std::complex<float>(0.0f, -0.5f) * (w15 - std::conj(w17));

	const float x16 = std::real<float>(w16);
	const float h16 = std::imag<float>(w16);

	f[ 0] = x0;
	f[ 1] = h0;
	f[ 2] = std::real<float>(x1);
	f[ 3] = std::real<float>(h1);
	f[ 4] = std::real<float>(x2);
	f[ 5] = std::real<float>(h2);
	f[ 6] = std::real<float>(x3);
	f[ 7] = std::real<float>(h3);
	f[ 8] = std::real<float>(x4);
	f[ 9] = std::real<float>(h4);
	f[10] = std::real<float>(x5);
	f[11] = std::real<float>(h5);
	f[12] = std::real<float>(x6);
	f[13] = std::real<float>(h6);
	f[14] = std::real<float>(x7);
	f[15] = std::real<float>(h7);
	f[16] = std::real<float>(x8);
	f[17] = std::real<float>(h8);
	f[18] = std::real<float>(x9);
	f[19] = std::real<float>(h9);
	f[20] = std::real<float>(x10);
	f[21] = std::real<float>(h10);
	f[22] = std::real<float>(x11);
	f[23] = std::real<float>(h11);
	f[24] = std::real<float>(x12);
	f[25] = std::real<float>(h12);
	f[26] = std::real<float>(x13);
	f[27] = std::real<float>(h13);
	f[28] = std::real<float>(x14);
	f[29] = std::real<float>(h14);
	f[30] = std::real<float>(x15);
	f[31] = std::real<float>(h15);
	f[32] = x16;
	f[33] = h16;
	f[34] = std::imag<float>(x1);
	f[35] = std::imag<float>(h1);
	f[36] = std::imag<float>(x2);
	f[37] = std::imag<float>(h2);
	f[38] = std::imag<float>(x3);
	f[39] = std::imag<float>(h3);
	f[40] = std::imag<float>(x4);
	f[41] = std::imag<float>(h4);
	f[42] = std::imag<float>(x5);
	f[43] = std::imag<float>(h5);
	f[44] = std::imag<float>(x6);
	f[45] = std::imag<float>(h6);
	f[46] = std::imag<float>(x7);
	f[47] = std::imag<float>(h7);
	f[48] = std::imag<float>(x8);
	f[49] = std::imag<float>(h8);
	f[50] = std::imag<float>(x9);
	f[51] = std::imag<float>(h9);
	f[52] = std::imag<float>(x10);
	f[53] = std::imag<float>(h10);
	f[54] = std::imag<float>(x11);
	f[55] = std::imag<float>(h11);
	f[56] = std::imag<float>(x12);
	f[57] = std::imag<float>(h12);
	f[58] = std::imag<float>(x13);
	f[59] = std::imag<float>(h13);
	f[60] = std::imag<float>(x14);
	f[61] = std::imag<float>(h14);
	f[62] = std::imag<float>(x15);
	f[63] = std::imag<float>(h15);
}
