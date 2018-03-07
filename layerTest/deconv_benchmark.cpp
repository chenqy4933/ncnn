#include <iostream>
#include <stdio.h>
#include "benchlayer.h"
#include "layer/arm/deconvolution_arm.h"
#include "layer/deconvolution.h"

using namespace std;

int main(int argc, char **argv)
{
    cout << "Benchmark_begin" << endl;
    ////////////////////////////////////////////生成对用的层
    const char *layer_name = "Deconvolution"; //在不同的平台下面生成对应的类，如在arm平台下面会生成Convolution_arm,在PC下面会生成Convolution_x86
        int circle_num = 100;
    BenchLayer *bench_conv = new BenchLayer(layer_name, circle_num);
    ///////////////////////////////////////////配置层的参数，根据各层里面定义的参数顺序来定义
    int kernel_size = 2;
    int stride = 2;
    int pad = 0;
    int out_channel = 8;
    bench_conv->paramdict_->set(0, out_channel);
    bench_conv->paramdict_->set(1, kernel_size); //kernel size
    bench_conv->paramdict_->set(3, stride);
    bench_conv->paramdict_->set(4, pad);                       //pad
    bench_conv->paramdict_->set(5, 1);                         //bias enable
    bench_conv->layer_->load_param(*(bench_conv->paramdict_)); //配置层里面的参数
    ////////////////////////////////////////////////生成数据
    ////////////输入数据
    int w = 400;
    int h = 400;
    int c = 8;
    Mat in_mat(w, h, c, sizeof(float));
    srand((unsigned)time(NULL));
    for (int p = 0; p < c; p++)
    {
        float *data = in_mat.channel(p);
        for (int i = 0; i < w * h; i++)
        {
            data[i] = rand() / (float)(RAND_MAX);
        }
    }
    /////////////////////权重，bias数据
    float *weight = new float[c * kernel_size * kernel_size * out_channel];
    float *bias = new float[c * out_channel];
    float *pweight = weight;
    for (int out = 0; out < out_channel; out++)
    {
        pweight = weight + c * kernel_size * kernel_size * out;
        for (int p = 0; p < c; p++)
        {
            pweight += p * kernel_size * kernel_size;
            for (int i = 0; i < kernel_size * kernel_size; i++)
            {
                //pweight[i]=1.0f;
                pweight[i] = rand() / (float)(RAND_MAX);
            }
            bias[out * c + p] = rand() / (float)(RAND_MAX);
        }
    }
    ///////////////////////////////////////////////配置层的weight等，首先需要将父类的指针转换成为子类的指针，才能调用子类的成员
    Deconvolution *player = dynamic_cast<Deconvolution *>(bench_conv->layer_);
    player->weight_data.data = weight;
    player->bias_data.data = bias;
    /////////////////////////////////////////////进行benchmark
    Mat out_mat;
    bench_conv->benchmark_layer(in_mat, out_mat);
    bench_conv->print_benchmark();
    ///////////////////////////////////////////释放内存
    if (NULL != weight)
    {
        delete[] weight;
        weight = NULL;
    }
    if (NULL != bias)
    {
        delete[] bias;
        bias = NULL;
    }
}