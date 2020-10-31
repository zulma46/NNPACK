#include <nnpack/fft.h>
#include <ref/fft/complex-ref.h>


void nnp_ifft8_dualreal__ref(const float* f, float* t) 
{
	const float x0 = f[0];
	const float h0 = f[1];
	const float x4 = f[8];
	const float h4 = f[9];

	const std::complex<float> x1 = std::complex<float>(f[2], f[10]);
	const std::complex<float> h1 = std::complex<float>(f[3], f[11]);
	const std::complex<float> x2 = std::complex<float>(f[4], f[12]);
	const std::complex<float> h2 = std::complex<float>(f[5], f[13]);
	const std::complex<float> x3 = std::complex<float>(f[6], f[14]);
	const std::complex<float> h3 = std::complex<float>(f[7], f[15]);

	std::complex<float> w0 = std::complex<float>(x0, h0);
	std::complex<float> w1 = x1 + std::complex<float>(0.0f, 1.0f) * h1;
	std::complex<float> w2 = x2 + std::complex<float>(0.0f, 1.0f) * h2;
	std::complex<float> w3 = x3 + std::complex<float>(0.0f, 1.0f) * h3;
	std::complex<float> w4 = std::complex<float>(x4, h4);
	std::complex<float> w5 = std::conj<float>(x3 - std::complex<float>(0.0f, 1.0f) * h3);
	std::complex<float> w6 = std::conj<float>(x2 - std::complex<float>(0.0f, 1.0f) * h2);
	std::complex<float> w7 = std::conj<float>(x1 - std::complex<float>(0.0f, 1.0f) * h1);
	
	ifft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	t[ 0] = std::real<float>(w0);
	t[ 1] = std::real<float>(w1);
	t[ 2] = std::real<float>(w2);
	t[ 3] = std::real<float>(w3);
	t[ 4] = std::real<float>(w4);
	t[ 5] = std::real<float>(w5);
	t[ 6] = std::real<float>(w6);
	t[ 7] = std::real<float>(w7);
	t[ 8] = std::imag<float>(w0);
	t[ 9] = std::imag<float>(w1);
	t[10] = std::imag<float>(w2);
	t[11] = std::imag<float>(w3);
	t[12] = std::imag<float>(w4);
	t[13] = std::imag<float>(w5);
	t[14] = std::imag<float>(w6);
	t[15] = std::imag<float>(w7);
}

void nnp_ifft16_dualreal__ref(const float* f, float* t) 
{
	const float x0 = f[0];
	const float h0 = f[1];
	const float x8 = f[16];
	const float h8 = f[17];

	const std::complex<float> x1 = std::complex<float>(f[ 2], f[18]);
	const std::complex<float> h1 = std::complex<float>(f[ 3], f[19]);
	const std::complex<float> x2 = std::complex<float>(f[ 4], f[20]);
	const std::complex<float> h2 = std::complex<float>(f[ 5], f[21]);
	const std::complex<float> x3 = std::complex<float>(f[ 6], f[22]);
	const std::complex<float> h3 = std::complex<float>(f[ 7], f[23]);
	const std::complex<float> x4 = std::complex<float>(f[ 8], f[24]);
	const std::complex<float> h4 = std::complex<float>(f[ 9], f[25]);
	const std::complex<float> x5 = std::complex<float>(f[10], f[26]);
	const std::complex<float> h5 = std::complex<float>(f[11], f[27]);
	const std::complex<float> x6 = std::complex<float>(f[12], f[28]);
	const std::complex<float> h6 = std::complex<float>(f[13], f[29]);
	const std::complex<float> x7 = std::complex<float>(f[14], f[30]);
	const std::complex<float> h7 = std::complex<float>(f[15], f[31]);

	std::complex<float> w0  = std::complex<float>(x0, h0);
	std::complex<float> w1  = x1 + std::complex<float>(0.0f, 1.0f) * h1;
	std::complex<float> w2  = x2 + std::complex<float>(0.0f, 1.0f) * h2;
	std::complex<float> w3  = x3 + std::complex<float>(0.0f, 1.0f) * h3;
	std::complex<float> w4  = x4 + std::complex<float>(0.0f, 1.0f) * h4;
	std::complex<float> w5  = x5 + std::complex<float>(0.0f, 1.0f) * h5;
	std::complex<float> w6  = x6 + std::complex<float>(0.0f, 1.0f) * h6;
	std::complex<float> w7  = x7 + std::complex<float>(0.0f, 1.0f) * h7;
	std::complex<float> w8  = std::complex<float>(x8, h8);
	std::complex<float> w9  = std::conj<float>(x7 - std::complex<float>(0.0f, 1.0f) * h7);
	std::complex<float> w10 = std::conj<float>(x6 - std::complex<float>(0.0f, 1.0f) * h6);
	std::complex<float> w11 = std::conj<float>(x5 - std::complex<float>(0.0f, 1.0f) * h5);
	std::complex<float> w12 = std::conj<float>(x4 - std::complex<float>(0.0f, 1.0f) * h4);
	std::complex<float> w13 = std::conj<float>(x3 - std::complex<float>(0.0f, 1.0f) * h3);
	std::complex<float> w14 = std::conj<float>(x2 - std::complex<float>(0.0f, 1.0f) * h2);
	std::complex<float> w15 = std::conj<float>(x1 - std::complex<float>(0.0f, 1.0f) * h1);
	
	ifft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	t[ 0] = std::real<float>(w0);
	t[ 1] = std::real<float>(w1);
	t[ 2] = std::real<float>(w2);
	t[ 3] = std::real<float>(w3);
	t[ 4] = std::real<float>(w4);
	t[ 5] = std::real<float>(w5);
	t[ 6] = std::real<float>(w6);
	t[ 7] = std::real<float>(w7);
	t[ 8] = std::real<float>(w8);
	t[ 9] = std::real<float>(w9);
	t[10] = std::real<float>(w10);
	t[11] = std::real<float>(w11);
	t[12] = std::real<float>(w12);
	t[13] = std::real<float>(w13);
	t[14] = std::real<float>(w14);
	t[15] = std::real<float>(w15);
	t[16] = std::imag<float>(w0);
	t[17] = std::imag<float>(w1);
	t[18] = std::imag<float>(w2);
	t[19] = std::imag<float>(w3);
	t[20] = std::imag<float>(w4);
	t[21] = std::imag<float>(w5);
	t[22] = std::imag<float>(w6);
	t[23] = std::imag<float>(w7);
	t[24] = std::imag<float>(w8);
	t[25] = std::imag<float>(w9);
	t[26] = std::imag<float>(w10);
	t[27] = std::imag<float>(w11);
	t[28] = std::imag<float>(w12);
	t[29] = std::imag<float>(w13);
	t[30] = std::imag<float>(w14);
	t[31] = std::imag<float>(w15);
}

void nnp_ifft32_dualreal__ref(const float* f, float* t) 
{
	const float x0 = f[0];
	const float h0 = f[1];
	const float x16 = f[32];
	const float h16 = f[33];

	const std::complex<float> x1  = std::complex<float>(f[ 2], f[34]);
	const std::complex<float> h1  = std::complex<float>(f[ 3], f[35]);
	const std::complex<float> x2  = std::complex<float>(f[ 4], f[36]);
	const std::complex<float> h2  = std::complex<float>(f[ 5], f[37]);
	const std::complex<float> x3  = std::complex<float>(f[ 6], f[38]);
	const std::complex<float> h3  = std::complex<float>(f[ 7], f[39]);
	const std::complex<float> x4  = std::complex<float>(f[ 8], f[40]);
	const std::complex<float> h4  = std::complex<float>(f[ 9], f[41]);
	const std::complex<float> x5  = std::complex<float>(f[10], f[42]);
	const std::complex<float> h5  = std::complex<float>(f[11], f[43]);
	const std::complex<float> x6  = std::complex<float>(f[12], f[44]);
	const std::complex<float> h6  = std::complex<float>(f[13], f[45]);
	const std::complex<float> x7  = std::complex<float>(f[14], f[46]);
	const std::complex<float> h7  = std::complex<float>(f[15], f[47]);
	const std::complex<float> x8  = std::complex<float>(f[16], f[48]);
	const std::complex<float> h8  = std::complex<float>(f[17], f[49]);
	const std::complex<float> x9  = std::complex<float>(f[18], f[50]);
	const std::complex<float> h9  = std::complex<float>(f[19], f[51]);
	const std::complex<float> x10 = std::complex<float>(f[20], f[52]);
	const std::complex<float> h10 = std::complex<float>(f[21], f[53]);
	const std::complex<float> x11 = std::complex<float>(f[22], f[54]);
	const std::complex<float> h11 = std::complex<float>(f[23], f[55]);
	const std::complex<float> x12 = std::complex<float>(f[24], f[56]);
	const std::complex<float> h12 = std::complex<float>(f[25], f[57]);
	const std::complex<float> x13 = std::complex<float>(f[26], f[58]);
	const std::complex<float> h13 = std::complex<float>(f[27], f[59]);
	const std::complex<float> x14 = std::complex<float>(f[28], f[60]);
	const std::complex<float> h14 = std::complex<float>(f[29], f[61]);
	const std::complex<float> x15 = std::complex<float>(f[30], f[62]);
	const std::complex<float> h15 = std::complex<float>(f[31], f[63]);

	std::complex<float> w0  = std::complex<float>(x0, h0);
	std::complex<float> w1  = x1  + std::complex<float>(0.0f, 1.0f) * h1;
	std::complex<float> w2  = x2  + std::complex<float>(0.0f, 1.0f) * h2;
	std::complex<float> w3  = x3  + std::complex<float>(0.0f, 1.0f) * h3;
	std::complex<float> w4  = x4  + std::complex<float>(0.0f, 1.0f) * h4;
	std::complex<float> w5  = x5  + std::complex<float>(0.0f, 1.0f) * h5;
	std::complex<float> w6  = x6  + std::complex<float>(0.0f, 1.0f) * h6;
	std::complex<float> w7  = x7  + std::complex<float>(0.0f, 1.0f) * h7;
	std::complex<float> w8  = x8  + std::complex<float>(0.0f, 1.0f) * h8;
	std::complex<float> w9  = x9  + std::complex<float>(0.0f, 1.0f) * h9;
	std::complex<float> w10 = x10 + std::complex<float>(0.0f, 1.0f) * h10;
	std::complex<float> w11 = x11 + std::complex<float>(0.0f, 1.0f) * h11;
	std::complex<float> w12 = x12 + std::complex<float>(0.0f, 1.0f) * h12;
	std::complex<float> w13 = x13 + std::complex<float>(0.0f, 1.0f) * h13;
	std::complex<float> w14 = x14 + std::complex<float>(0.0f, 1.0f) * h14;
	std::complex<float> w15 = x15 + std::complex<float>(0.0f, 1.0f) * h15;
	std::complex<float> w16 = std::complex<float>(x16, h16);
	std::complex<float> w17 = std::conj<float>(x15 - std::complex<float>(0.0f, 1.0f) * h15);
	std::complex<float> w18 = std::conj<float>(x14 - std::complex<float>(0.0f, 1.0f) * h14);
	std::complex<float> w19 = std::conj<float>(x13 - std::complex<float>(0.0f, 1.0f) * h13);
	std::complex<float> w20 = std::conj<float>(x12 - std::complex<float>(0.0f, 1.0f) * h12);
	std::complex<float> w21 = std::conj<float>(x11 - std::complex<float>(0.0f, 1.0f) * h11);
	std::complex<float> w22 = std::conj<float>(x10 - std::complex<float>(0.0f, 1.0f) * h10);
	std::complex<float> w23 = std::conj<float>(x9  - std::complex<float>(0.0f, 1.0f) * h9);
	std::complex<float> w24 = std::conj<float>(x8  - std::complex<float>(0.0f, 1.0f) * h8);
	std::complex<float> w25 = std::conj<float>(x7  - std::complex<float>(0.0f, 1.0f) * h7);
	std::complex<float> w26 = std::conj<float>(x6  - std::complex<float>(0.0f, 1.0f) * h6);
	std::complex<float> w27 = std::conj<float>(x5  - std::complex<float>(0.0f, 1.0f) * h5);
	std::complex<float> w28 = std::conj<float>(x4  - std::complex<float>(0.0f, 1.0f) * h4);
	std::complex<float> w29 = std::conj<float>(x3  - std::complex<float>(0.0f, 1.0f) * h3);
	std::complex<float> w30 = std::conj<float>(x2  - std::complex<float>(0.0f, 1.0f) * h2);
	std::complex<float> w31 = std::conj<float>(x1  - std::complex<float>(0.0f, 1.0f) * h1);

	ifft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	t[ 0] = std::real<float>(w0);
	t[ 1] = std::real<float>(w1);
	t[ 2] = std::real<float>(w2);
	t[ 3] = std::real<float>(w3);
	t[ 4] = std::real<float>(w4);
	t[ 5] = std::real<float>(w5);
	t[ 6] = std::real<float>(w6);
	t[ 7] = std::real<float>(w7);
	t[ 8] = std::real<float>(w8);
	t[ 9] = std::real<float>(w9);
	t[10] = std::real<float>(w10);
	t[11] = std::real<float>(w11);
	t[12] = std::real<float>(w12);
	t[13] = std::real<float>(w13);
	t[14] = std::real<float>(w14);
	t[15] = std::real<float>(w15);
	t[16] = std::real<float>(w16);
	t[17] = std::real<float>(w17);
	t[18] = std::real<float>(w18);
	t[19] = std::real<float>(w19);
	t[20] = std::real<float>(w20);
	t[21] = std::real<float>(w21);
	t[22] = std::real<float>(w22);
	t[23] = std::real<float>(w23);
	t[24] = std::real<float>(w24);
	t[25] = std::real<float>(w25);
	t[26] = std::real<float>(w26);
	t[27] = std::real<float>(w27);
	t[28] = std::real<float>(w28);
	t[29] = std::real<float>(w29);
	t[30] = std::real<float>(w30);
	t[31] = std::real<float>(w31);
	t[32] = std::imag<float>(w0);
	t[33] = std::imag<float>(w1);
	t[34] = std::imag<float>(w2);
	t[35] = std::imag<float>(w3);
	t[36] = std::imag<float>(w4);
	t[37] = std::imag<float>(w5);
	t[38] = std::imag<float>(w6);
	t[39] = std::imag<float>(w7);
	t[40] = std::imag<float>(w8);
	t[41] = std::imag<float>(w9);
	t[42] = std::imag<float>(w10);
	t[43] = std::imag<float>(w11);
	t[44] = std::imag<float>(w12);
	t[45] = std::imag<float>(w13);
	t[46] = std::imag<float>(w14);
	t[47] = std::imag<float>(w15);
	t[48] = std::imag<float>(w16);
	t[49] = std::imag<float>(w17);
	t[50] = std::imag<float>(w18);
	t[51] = std::imag<float>(w19);
	t[52] = std::imag<float>(w20);
	t[53] = std::imag<float>(w21);
	t[54] = std::imag<float>(w22);
	t[55] = std::imag<float>(w23);
	t[56] = std::imag<float>(w24);
	t[57] = std::imag<float>(w25);
	t[58] = std::imag<float>(w26);
	t[59] = std::imag<float>(w27);
	t[60] = std::imag<float>(w28);
	t[61] = std::imag<float>(w29);
	t[62] = std::imag<float>(w30);
	t[63] = std::imag<float>(w31);
}
