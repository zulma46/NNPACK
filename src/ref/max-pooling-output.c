#include <nnpack.h>
#include <nnpack/reference.h>
#include <nnpack/utils.h>
#ifdef _WIN32
#include <math.h>
#endif

struct max_pooling_output_context {
	size_t channels;
	struct nnp_size input_size;
	struct nnp_padding input_padding;
	struct nnp_size pooling_size;
	struct nnp_size pooling_stride;
	struct nnp_size output_size;
	const float* input;
	float* output;
};

static void compute_max_pooling_output(
#ifdef _WIN32
	const struct max_pooling_output_context* context,
#else
	const struct max_pooling_output_context context[restrict static 1],
#endif
	size_t sample, size_t channel)
{
	const size_t channels                  = context->channels;
	const struct nnp_size input_size       = context->input_size;
	const struct nnp_padding input_padding = context->input_padding;
	const struct nnp_size pooling_size     = context->pooling_size;
	const struct nnp_size pooling_stride   = context->pooling_stride;
	const struct nnp_size output_size      = context->output_size;

#ifdef _WIN32
	const float* input = context->input;
	float* output = context->output;
#else
	const float (*input)[channels][input_size.height][input_size.width] =
		(const float(*)[channels][input_size.height][input_size.width]) context->input;
	float (*output)[channels][output_size.height][output_size.width] =
		(float(*)[channels][output_size.height][output_size.width]) context->output;
#endif

	for (size_t y = 0; y < output_size.height; y++) {
		for (size_t x = 0; x < output_size.width; x++) {
#ifdef _WIN32
			float v = -INFINITY;
#else
			float v = -__builtin_inff();
#endif
			for (size_t i = 0; i < pooling_size.height; i++) {
				const size_t s = y * pooling_stride.height + i - input_padding.top;
				if (s < input_size.height) {
					for (size_t j = 0; j < pooling_size.width; j++) {
						const size_t t = x * pooling_stride.width + j - input_padding.left;
						if (t < input_size.width) {
#ifdef _WIN32
							v = maxf(input[(sample * channels * input_size.width * input_size.height) + (channel * input_size.width * input_size.height) + (s * input_size.width) + t], v);
#else
							v = maxf(input[sample][channel][s][t], v);
#endif
						}
					}
				}
			}
#ifdef _WIN32
			output[(sample * channels * output_size.width * output_size.height) + (channel * output_size.width * output_size.height) + (y * output_size.width) + x] = v;
#else
			output[sample][channel][y][x] = v;
#endif
		}
	}
}

void nnp_max_pooling_output__reference(
	size_t batch_size,
	size_t channels,
	struct nnp_size input_size,
	struct nnp_padding input_padding,
	struct nnp_size pooling_size,
	struct nnp_size pooling_stride,
	const float* input,
	float* output,
	pthreadpool_t threadpool)
{
	const struct nnp_size output_size = {
		.height = divide_round_up(doz(input_padding.top + input_size.height + input_padding.bottom, pooling_size.height), pooling_stride.height) + 1,
		.width = divide_round_up(doz(input_padding.left + input_size.width + input_padding.right, pooling_size.width), pooling_stride.width) + 1,
	};

	struct max_pooling_output_context max_pooling_output_context = {
		.channels = channels,
		.input_size = input_size,
		.input_padding = input_padding,
		.pooling_size = pooling_size,
		.pooling_stride = pooling_stride,
		.output_size = output_size,
		.input = input,
		.output = output
	};

	pthreadpool_parallelize_2d(threadpool,
		(pthreadpool_function_2d_t) compute_max_pooling_output,
		&max_pooling_output_context,
		batch_size, channels,
		PTHREADPOOL_FLAG_DISABLE_DENORMALS);
}
