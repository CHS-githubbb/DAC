

#include "pch.h"
#include "Layers.h"
#include "Layers.cpp"
#include <iostream>
using namespace Layers;



typedef double dtype;


extern dtype features_0_weight_t[1][1][3][3],
				features_0_bias_t[1];


dtype test_input[1][1][5][5];

dtype test_output[1][1][3][3];


/*
int main() {
	std::cout << "link!\n";
	Conv<dtype, 1, 3, 1, 1, 5, 5> conv(features_0_weight_t, features_0_bias_t);


	for (int i = 0; i < 1; ++i) {
		for (int j = 0; j < 1; ++j) {
			for (int m = 0; m < 5; ++m) {
				for (int n = 0; n < 5; ++n) {
					test_input[i][j][m][n] = 1;
				}
			}
		}
	}


	conv(test_input, test_output);

	for (int i = 0; i < 1; ++i) {
		for (int j = 0; j < 1; ++j) {
			for (int m = 0; m < 3; ++m) {
				for (int n = 0; n < 3; ++n) {
					std::cout << test_output[i][j][m][n] << " ";
				}

				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}
*/