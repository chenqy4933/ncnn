
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "layer/convolution.h"

using namespace ncnn;
using namespace std;
/*
forward - pass:
    [input]                             [kernel]   [stride,dilation,pad]   [bias]
 
                                                 
[0.5  0.5  1.5  1.5  2.5  2.5                                                       [ 2.5f,  6.5f, 10.5f,
 0.5  0.5  1.5  1.5  2.5  2.5           [1,1,                                         6.5f, 10.5f, 14.5f,
 1.5  1.5  2.5  2.5  3.5  3.5       *   1,1]        [2,1,0]          +   0.5     =   10.5f, 14.5f, 18.5f] 
 1.5  1.5  2.5  2.5  3.5  3.5
 2.5  2.5  3.5  3.5  4.5  4.5
 2.5  2.5  3.5  3.5  4.5  4.5 ]
 
 
 [0.0f, 1.0f, 2.0f, 3.0f,           [1,1,                                         [4.5  12.5
  1.0f, 2.0f, 3.0f, 4.0f,       *    1,1]        [2,1,0]          +   0.5     =    12.5 20.5 ]
  2.0f, 3.0f, 4.0f, 5.0f,
  3.0f, 4.0f, 5.0f, 6.0f]

 
 [0.0f, 1.0f, 2.0f,     [1,1,                                          [ 0.5  1.5  3.5  2.5
 1.0f, 2.0f, 3.0f,   *   1,1]        [1,1,0]         +   0.5     =       1.5  4.5  8.5  5.5
 2.0f, 3.0f, 4.0f]                                                       3.5  8.5  12.5 7.5
                                                                         2.5  5.5  7.5  4.5 ]
 
 [0.0f, 1.0f, 2.0f,     [1,1,                                          [ 0.5  1.5  2.5  1.5  2.5
 1.0f, 2.0f, 3.0f,   *   1,1]        [1,2,0]         +   0.5     =       1.5  2.5  4.5  2.5  3.5
 2.0f, 3.0f, 4.0f]                                                       2.5  4.5  8.5  4.5  6.5
                                                                         1.5  2.5  4.5  2.5  3.5
                                                                         2.5  3.5  6.5  3.5  4.5 ]

 */

#define CHANNEL 1
#define KERNEL 2
#define DILATION 1
#define NUMOUT 2

int main(int argc, char **argv)
{
    //./Test stride pad
    // layer params
    int stride  = 2;
    int pad     = 0;
    
    if (argc > 1 ) {
		istringstream Tmpstream(argv[1]);
        Tmpstream >> stride;
	}
    if (argc > 2 ) {
		istringstream Tmpstream(argv[2]);
        Tmpstream >> pad;
	}
    
    Convolution convolution_layer;
    convolution_layer.num_output = NUMOUT;
    convolution_layer.kernel_w = KERNEL;
    convolution_layer.kernel_h = KERNEL;
    convolution_layer.dilation_w = DILATION;
	convolution_layer.dilation_h = DILATION;
    convolution_layer.stride_w = stride;
	convolution_layer.stride_h = stride;
    convolution_layer.pad_w = pad;
	convolution_layer.pad_h = pad;
    convolution_layer.bias_term = 1;
    convolution_layer.weight_data_size = NUMOUT * CHANNEL *KERNEL * KERNEL;

    // input & output
    float in2X2[] = {
        0.0f, 1.0f,
        2.0f, 3.0f,
        
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    
    float in3X3[] = {
        0.0f, 1.0f, 2.0f,
        1.0f, 2.0f, 3.0f,
        2.0f, 3.0f, 4.0f,
        0.f,0.f,0.f,
        
        1.0f, 2.0f, 3.0f,
        2.0f, 3.0f, 4.0f,
        3.0f, 4.0f, 5.0f,
        0.f,0.f,0.f
    };
    float in4X4[] = {
        0.0f, 1.0f, 2.0f, 3.0f,
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,

        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    };
    float in5X5[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        0.0f, 0.0f, 0.0f,
        
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        0.0f, 0.0f, 0.0f
    };

    // weights & bias
    float w2[] = {
        0.5f, 0.5f, 0.5f, 1.0f,
        0.5f, 0.5f, 0.5f, 1.0f,
        0.5f, 0.5f, 0.5f, 1.0f,
        0.5f, 0.5f, 0.5f, 1.0f,
        
        0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f
    };

    float w[] = {
        1.0f, 1.0f,
        1.0f, 1.0f,
        
        1.0f, 1.0f,
        1.0f, 1.0f
    };
    
    float w1[] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    };
    
    float b[] = {
        0.5f,
        1.0f
    };
 
    // forward
    float *in = in4X4;
    int input_hw = 4;
    Mat mat_in(input_hw, input_hw, CHANNEL, in);
    Mat mat_out;

    convolution_layer.bias_data.data = b;
    convolution_layer.weight_data.data = w;
    convolution_layer.forward(mat_in, mat_out);
    //convolution_layer.forward_eigen(mat_in, mat_out);
    // check expect
    printf("w: %d.\n", mat_out.w);
    printf("h: %d.\n", mat_out.h);
    printf("c: %d.\n", mat_out.c);
    for (int n=0; n<mat_out.c; n++) {
        Mat mat_tmp = mat_out.channel(n);
        printf("channel: %d:\n", n);
        for (int i = 0; i < (int)mat_tmp.h; ++i)
        {
            for (int j = 0; j < (int)mat_tmp.w; ++j)
                printf("%-4.1f ", mat_tmp[i*mat_tmp.w + j]);
            printf("\n");
        }
    }
    printf("\n");
    return 0;
}
