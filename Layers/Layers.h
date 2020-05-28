#ifndef LAYERS_H
#define LAYERS_H


namespace Layers {

	//declaration of kernel
	template<class dtype, unsigned kchannel, unsigned ksize>
	class Kernel {
	private:
		unsigned kernel_size;
		unsigned kernel_channel;
		dtype weights[kchannel][ksize][ksize];
		dtype bias;

	public:
		Kernel();

		Kernel(dtype weights_[kchannel][ksize][ksize], dtype bias_);

		dtype*** GetWeight();

		void SetWeight(dtype weights_[kchannel][ksize][ksize]);

		dtype GetBias();

		void SetBias(dtype bias_);

		unsigned GetSize();

		unsigned GetNumOfChannel();

		dtype At(int channel, int height, int width);
	};






	//declaration of conv
	template<class dtype, 
				unsigned out_channel,
				unsigned kernel_size,
				unsigned batch_size,
				unsigned in_channel,
				unsigned height,
				unsigned width,
				unsigned padding = 0,
				unsigned stride = 1>
	class Conv {
	private:
		Kernel<dtype, in_channel, kernel_size> kernels[out_channel];

	public:

		Conv() {};

		Conv(dtype weights_[out_channel][in_channel][kernel_size][kernel_size], dtype bias_[out_channel]);

		Kernel<dtype, in_channel, kernel_size>& GetKernel(unsigned index);

		//run the conv-layer
		void operator () (
			dtype input [batch_size][in_channel][height][width],
			dtype output[batch_size][out_channel]
						[((height - kernel_size) + 2 * padding) / stride + 1]
						[((width - kernel_size) + 2 * padding) / stride + 1]
			);
	};






	//declaration of fc
	template<class dtype,
		unsigned out_channel,
		unsigned batch_size,
		unsigned in_channel,
		unsigned height,
		unsigned width>
	class FullConnect {
	private:
		Kernel<dtype, in_channel, height> kernels[out_channel];

	public:

		FullConnect() {};

		FullConnect(dtype weights_[out_channel][in_channel][height][width], dtype bias_[out_channel]);

		Kernel<dtype, in_channel, height>& GetKernel(unsigned index);

		void operator()(
			dtype input[batch_size][in_channel][height][width],
			dtype output[batch_size][out_channel][1][1]
			);
	};







	//declaration of pool-layer
	template<class dtype,
			unsigned kernel_size,
			unsigned batch_size,
			unsigned in_channel,
			unsigned height,
			unsigned width,
			unsigned stride = 1>
		void Pool(
			dtype input[batch_size][in_channel][height][width],
			dtype output[batch_size][in_channel][(height - kernel_size) / stride + 1][(width - kernel_size) / stride + 1],
			bool(*func)(dtype, dtype)
		);


	template<class dtype,
		unsigned kernel_size,
		unsigned batch_size,
		unsigned in_channel,
		unsigned height,
		unsigned width,
		unsigned stride = 1>
		void MaxPool(
			dtype input[batch_size][in_channel][height][width],
			dtype output[batch_size][in_channel][(height - kernel_size) / stride + 1][(width - kernel_size) / stride + 1]
		);








	//declaration of activate function
	template<class dtype,
		unsigned batch_size,
		unsigned in_channel,
		unsigned height,
		unsigned width>
		void Activate(
			dtype input[batch_size][in_channel][height][width],
			void(*specified_activate_func)(dtype&)
		);



	template<class dtype,
		unsigned batch_size,
		unsigned in_channel,
		unsigned height,
		unsigned width>
		void ReLU(
			dtype input[batch_size][in_channel][height][width]
		);



	template<class dtype,
		unsigned batch_size,
		unsigned in_channel,
		unsigned height,
		unsigned width>
		void Sigmoid(
			dtype input[batch_size][in_channel][height][width]
		);
}

#endif

