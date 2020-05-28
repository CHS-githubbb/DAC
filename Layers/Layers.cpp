#include "pch.h"
#include "Layers.h"
#include<cstring>
#include<iostream>
#include<cmath>


//implementation of stuffs in Layers
namespace Layers {


/**********************************************************
implementation of Kernel

@template parameters:

	dtype:			compute type
	kchannel:		num of channels of kernel
	ksize:			size of kernel

***********************************************************/

/*
@description: kernel constructor, weights initialized wiht 0s
*/
template<class dtype, unsigned kchannel, unsigned ksize>
	Kernel<dtype, kchannel, ksize>::
		Kernel(): kernel_size(ksize), kernel_channel(kchannel), bias(0)
	{
		memset(weights, 0, sizeof(weights));
	}








/*
@description: kernel constructor with parameters
@parameter:
	weights_:		initial weights
	bias_:			initial bias
*/
template<class dtype, unsigned kchannel, unsigned ksize>
	Kernel<dtype, kchannel, ksize>::
		Kernel(dtype weights_[kchannel][ksize][ksize], dtype bias_) :bias(bias_) {
			memcpy(this->weights, weights_, sizeof(weights_));
		}






/*
@description: return the weights of the kernel
*/
template<class dtype, unsigned kchannel, unsigned ksize>
dtype*** Kernel<dtype, kchannel, ksize>::
	GetWeight()
	{
		return this->weights;
	}





/*
@description: set the weights of kernel
@parameter: a dtype*** which saves the weights(3 dimensions) that you want to set
*/
template<class dtype, unsigned kchannel, unsigned ksize>
void Kernel<dtype, kchannel, ksize>::
	SetWeight(dtype weights_[kchannel][ksize][ksize] )
	{
		memcpy(weights, weights_, sizeof(weights));

		/*
		for (int i = 0; i < kchannel; ++i) {
			for (int j = 0; j < ksize; ++j) {
				for (int k = 0; k < ksize; ++k) {
					std::cout << weights[i][j][k] << " ";
				}

				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		*/

	}






/*
@description: return the bias of the kernel
*/
template<class dtype, unsigned kchannel, unsigned ksize>
dtype Kernel<dtype, kchannel, ksize>::
	GetBias() {
		return this->bias;
	}




/*
@description:	set the bias of kernel
@parameter:		the bias you want to set
*/
template<class dtype, unsigned kchannel, unsigned ksize>
void Kernel<dtype, kchannel, ksize>::
	SetBias(dtype bias_) {
		this->bias = bias_;
	}





/*
@description: return the size(height or weight) of the kernel
*/
template<class dtype, unsigned kchannel, unsigned ksize>
unsigned Kernel<dtype, kchannel, ksize>::
	GetSize() {
		return this->kernel_size;
	}





/*
@description: return the number of channels of the kernel
*/
template<class dtype, unsigned kchannel, unsigned ksize>
unsigned Kernel<dtype, kchannel, ksize>::
	GetNumOfChannel() {
		return this->kernel_channel;
	}







/*
@description: get a element of the kernel at [width][height][channel]
@parameter1: weights[x][x][channel]
@parameter2: weights[x][height][x]
@parameter3: weights[x][x][width]
*/
template<class dtype, unsigned kchannel, unsigned ksize>
dtype Kernel<dtype, kchannel, ksize>::
	At(int channel, int height, int width)
	{
		return weights[channel][height][width];
	}







/**********************************************************
implementation of Conv

@template parameters:

	dtype:			compute type
	out_channel:	num of channels of output data
	ksize:			size of kernel
	batch_size:		batch_size of input data
	in_channel:		num of channels of input data
	height:			height of input data
	width:			width of input data
	padding:		nothing but decoration
	stride:			set the stride, default by 1

***********************************************************/







/*
@description: conv constructor with parameters
@parameter:
	weights_:		initial weights
	bias_:			initial bias
*/
template<class dtype,
	unsigned out_channel,
	unsigned kernel_size,
	unsigned batch_size,
	unsigned in_channel,
	unsigned height,
	unsigned width,
	unsigned padding,
	unsigned stride>
		Conv<dtype,
		out_channel,
		kernel_size,
		batch_size,
		in_channel,
		height,
		width,
		padding,
		stride>::
	Conv(dtype weights_[out_channel][in_channel][kernel_size][kernel_size],
		dtype bias_[out_channel]) {
			for (unsigned oc = 0; oc < out_channel; ++oc) {
				Kernel<dtype, in_channel, kernel_size>& ith_kernel = GetKernel(oc);
				ith_kernel.SetWeight(weights_[oc]);
				ith_kernel.SetBias(bias_[oc]);

				/*
				std::cout << "check kernel" << std::endl;
				for (int i = 0; i < in_channel; ++i) {
					for (int j = 0; j < kernel_size; ++j) {
						for (int k = 0; k < kernel_size; ++k) {
							std::cout << ith_kernel.At(i, j, k) << " ";
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
				}
				*/
				

			}
		}





/*
@description: return the the ith kernel of the conv-layer
@parameter:  index, the ith kernel you need, kernels[index]
*/
template<class dtype,
	unsigned out_channel,
	unsigned kernel_size,
	unsigned batch_size,
	unsigned in_channel,
	unsigned height,
	unsigned width,
	unsigned padding,
	unsigned stride>
Kernel<dtype, in_channel, kernel_size>&
	Conv<dtype, 
			out_channel,
			kernel_size,
			batch_size,
			in_channel,
			height,
			width,
			padding,
			stride>::
		GetKernel(unsigned index)
		{
			return this->kernels[index];
		}




/*
@description:run the conv-layer:
	input arranged in (batch_size, in_channel, height, width)
	output arranged in (batch_size, out_channel, height, width)


@parameter:
	input:		input data
	output:		output data

*/
template<class dtype,
	unsigned out_channel,
	unsigned kernel_size,
	unsigned batch_size,
	unsigned in_channel,
	unsigned height,
	unsigned width,
	unsigned padding,
	unsigned stride>
void Conv<dtype,
			out_channel,
			kernel_size,
			batch_size,
			in_channel,
			height,
			width,
			padding,
			stride>::
			operator () (
				dtype input[batch_size][in_channel][height][width],
				dtype output[batch_size][out_channel]
							[((height - kernel_size) + 2 * padding) / stride + 1]
							[((width - kernel_size) + 2 * padding) / stride + 1]
				)
		{
			
			
			//check the input
			/*
			std::cout << "check input!" << std::endl;
			//loop for input_batch
			for (int b = 0; b < batch_size; ++b) {
				//loop for input_channel
				for (int ic = 0; ic < in_channel; ++ic) {
					//loop for input_height
					for (int ih = 0; ih < height; ++ih) {
						//loop for input_weight
						for (int iw = 0; iw < width; ++iw) {
							std::cout << input[b][ic][ih][iw] << " ";
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
			std::cout << "end check input!" << std::endl;
			*/
			
					
			
			
			//loop for batch_size
			for (unsigned b = 0; b < batch_size; ++b) {
				//loop for out_channel	(actually the index of kernel)
				for (unsigned oc = 0; oc < out_channel; ++oc) {

					dtype bias = GetKernel(oc).GetBias();

					//loop for input_height
					for (int ih = -int(padding); ih < int(height - kernel_size + padding + 1); ih += stride) {
						//loop for input_weight
						for (int iw = -int(padding); iw < int(width - kernel_size + padding + 1); iw += stride) {

							dtype sum = 0;

							//loop for input_channel
							for (unsigned ic = 0; ic < in_channel; ++ic) {
								//loop for kernel_height
								for (unsigned kh = 0; kh < kernel_size; ++kh) {
									//loop for kernel width
									for (unsigned kw = 0; kw < kernel_size; ++kw) {
										if (ih + kh < 0 || iw + kw < 0 ||
											ih + kh >= height || iw + kw >= width) {
											sum += 0;
										}
										else {
											//std::cout << "input: " << input[b][ic][ih + kh][iw + kw] << std::endl;
											sum += input[b][ic][ih + kh][iw + kw] * GetKernel(oc).At(ic, kh, kw);
										}
									}
								}
							}
							//accumulate
							output[b][oc][(ih + padding) / stride][(iw + padding) / stride] = sum + bias;
						}
					}
				}
			}				
		}








/**********************************************************
implementation of FullConnect

@template parameters:

	dtype:			compute type
	out_channel:	num of channels of output data
	batch_size:		batch_size of input data
	in_channel:		num of channels of input data
	height:			height of input data
	width:			width of input data

***********************************************************/







/*
@description: fc constructor with parameters
@parameter:
	weights_:		initial weights
	bias_:			initial bias
*/
template<class dtype,
	unsigned out_channel,
	unsigned batch_size,
	unsigned in_channel,
	unsigned height,
	unsigned width>
		FullConnect<dtype,
		out_channel,
		batch_size,
		in_channel,
		height,
		width>::
	FullConnect(dtype weights_[out_channel][in_channel][height][width],
		dtype bias_[out_channel]) {

				for (unsigned oc = 0; oc < out_channel; ++oc) {
					Kernel<dtype, in_channel, height>& ith_kernel = GetKernel(oc);
					ith_kernel.SetWeight(weights_[oc]);
					ith_kernel.SetBias(bias_[oc]);
				}
		}





/*
@description: return the kernel of the fc-layer
*/
template<class dtype,
	unsigned out_channel,
	unsigned batch_size,
	unsigned in_channel,
	unsigned height,
	unsigned width>
Kernel<dtype, in_channel, height>& 
			FullConnect<dtype,
						out_channel,
						batch_size,
						in_channel,
						height,
						width>::
						GetKernel(unsigned index) {
							return kernels[index];
						}





/*
@description:run the fc-layer:
	input arranged in (batch_size, in_channel, height, width)
	output arranged in (batch_size, out_channel, height, width)


@parameter:
	input:		input data
	output:		output data
*/
template<class dtype,
	unsigned out_channel,
	unsigned batch_size,
	unsigned in_channel,
	unsigned height,
	unsigned width>
void FullConnect<dtype,
					out_channel,
					batch_size,
					in_channel,
					height,
					width>::
					operator() (
						dtype input[batch_size][in_channel][height][width],
						dtype output[batch_size][out_channel][1][1]
						) {
							//loop for batch_size
							for (unsigned b = 0; b < batch_size; ++b) {
								//loop for out_channel	(actually the index of kernel)
								for (unsigned oc = 0; oc < out_channel; ++oc) {

									dtype bias = GetKernel(oc).GetBias();
									dtype sum = 0;

									//loop for input_channel
									for (unsigned ic = 0; ic < in_channel; ++ic) {
										//loop for kernel_height
										for (unsigned kh = 0; kh < height; ++kh) {
											//loop for kernel width
											for (unsigned kw = 0; kw < width; ++kw) {
												//std::cout << "input: " << input[b][ic][ih + kh][iw + kw] << std::endl;
												sum += input[b][ic][kh][kw] * GetKernel(oc).At(ic, kh, kw);
											}
										}
									}
									//accumulate
									output[b][oc][0][0] = sum + bias;
								}
							}
						}











/**********************************************************
implementation of pool-layer

@template parameters:

	dtype:			compute type
	ksize:			size of kernel
	batch_size:		batch_size of input data
	in_channel:		num of channels of input data
	height:			height of input data
	width:			width of input data
	padding:		nothing but decoration
	stride:			set the stride, default by 1

***********************************************************/




/*
@description:	template of pool-layer
	input arranged in (batch_size, in_channel, height, width)
	output arranged in (batch_size, out_channel, height, width)

@parameter:
	input:		input data
	output:		output data
	func:		a function pointer which define the rule of the filter
*/
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
	) {
		//loop for batch_size
		for (unsigned b = 0; b < batch_size; ++b) {
			//loop for input_channel
			for (unsigned ic = 0; ic < in_channel; ++ic) {
				//loop for input_height
				for (unsigned ih = 0; ih < height - kernel_size + 1; ih += stride) {
					//loop for input_weight
					for (unsigned iw = 0; iw < width - kernel_size + 1; iw += stride) {

						dtype winner = input[b][ic][ih][iw];

						//loop for kernel_height
						for (unsigned kh = 0; kh < kernel_size; ++kh) {
							//loop for kernel_width
							for (unsigned kw = 0; kw < kernel_size; ++kw) {
								if (ih + kh >= height || iw + kw >= width) {
									//out of loop
								}
								else {
									winner = func(winner, input[b][ic][ih + kh][iw + kw]) ?
										winner : input[b][ic][ih + kh][iw + kw];
								}
							}
						}
						//get it!
						output[b][ic][ih / stride][iw / stride] = winner;
					}
				}
			}
		}
	}




/*
@description:	implementation of max-filter
*/
template<class dtype>
bool max_filter(dtype x, dtype y) {
	return x > y;
}





/*
@description:	implementation of max_pool-layer
	input arranged in (batch_size, in_channel, height, width)
	output arranged in (batch_size, out_channel, height, width)

@parameter:
	input:		input data
	output:		output data
*/
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
	) {
		//run the Pool function with max_filter
		Pool<dtype, 
			kernel_size, 
			batch_size,
			in_channel,
			height,
			width,
			stride>
			(input, output, max_filter<dtype>);
	}












/**********************************************************
implementation of activate function

@template parameters:

	dtype:			compute type
	batch_size:		batch_size of input data
	in_channel:		num of channels of input data
	height:			height of input data
	width:			width of input data

***********************************************************/


/*
@description:	template of activate function
	input arranged in (batch_size, in_channel, height, width)

@parameter:
	input:							input data
	specified_activate_func:		a function pointer which define the rule of the activation
*/
template<class dtype,
	unsigned batch_size,
	unsigned in_channel,
	unsigned height,
	unsigned width>
	void Activate(
		dtype input[batch_size][in_channel][height][width],
		void(*specified_activate_func)(dtype&)
	) {
		//loop for batch_size
		for (unsigned b = 0; b < batch_size; ++b) {
			//loop for input_channel
			for (unsigned ic = 0; ic < in_channel; ++ic) {
				//loop for input_height
				for (unsigned ih = 0; ih < height; ++ih) {
					//loop for input_weight
					for (unsigned iw = 0; iw < width; ++iw) {
						specified_activate_func(input[b][ic][ih][iw]);
					}
				}
			}
		}
		
	}





/*
@description:	implementation of relu activate function
*/
template<class dtype>
void relu_activate_func(dtype& x) {
	x = x >= 0 ? x : 0;
}







/*
@description:	implementation of ReLU
	input arranged in (batch_size, in_channel, height, width)

@parameter:
	input:		input data

*/
template<class dtype,
	unsigned batch_size,
	unsigned in_channel,
	unsigned height,
	unsigned width>
	void ReLU(
		dtype input[batch_size][in_channel][height][width]
	) {
			Activate<dtype,
				batch_size,
				in_channel,
				height,
				width>(input, relu_activate_func);
	}





/*
@description:	implementation of sigmoid activate function
*/
template<class dtype>
void sigmoid_activate_func(dtype& x) {
	x = 1.0 / 1.0 + exp(-x);
}




/*
@description:	implementation of Sigmoid
	input arranged in (batch_size, in_channel, height, width)

@parameter:
	input:		input data

*/
template<class dtype,
	unsigned batch_size,
	unsigned in_channel,
	unsigned height,
	unsigned width>
	void Sigmoid(
		dtype input[batch_size][in_channel][height][width]
	) {
			Activate<dtype,
				batch_size,
				in_channel,
				height,
				width>(input, sigmoid_activate_func);
	}

}










