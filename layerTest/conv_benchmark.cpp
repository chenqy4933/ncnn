#include <iostream>
#include <stdio.h>
#include "benchlayer.h"
#include "layer/arm/convolution_arm.h"
#include "layer/x86/convolution_x86.h"

using namespace std;

int main(int argc,char** argv)
{
    cout<<"Benchmark_begin"<<endl;
    bool benchmark=false;
    ////////////////////////////////////////////生成对用的层
    const char* layer_name="Convolution" ;  //在不同的平台下面生成对应的类，如在arm平台下面会生成Convolution_arm,在PC下面会生成Convolution_x86
    int circle_num=20;
    BenchLayer *bench_conv = new BenchLayer(layer_name,circle_num);
    ///////////////////////////////////////////配置层的参数，根据各层里面定义的参数顺序来定义
    int kernel_size=2;
    int stride=2;
    int pad=0;
    int out_channel = 1;
    bench_conv->paramdict_->set(0, out_channel);
    bench_conv->paramdict_->set(1, kernel_size); //kernel size
    bench_conv->paramdict_->set(3, stride);
    bench_conv->paramdict_->set(4, pad);         //pad
    bench_conv->paramdict_->set(5,1);            //bias enable
    bench_conv->layer_->load_param(*(bench_conv->paramdict_)); //配置层里面的参数
    ////////////////////////////////////////////////生成数据
             ////////////输入数据
    int w=18;
    int h=18;
    int c=3;
    Mat in_mat(w,h,c,sizeof(float));
    srand((unsigned)time(NULL));
    for(int p=0;p<c;p++)
    {
        float* data=in_mat.channel(p);
        for(int i=0;i<w*h;i++)
        {
            data[i]=i+p;
            //data[i] = rand()/(float)(RAND_MAX);
        }
    }
            /////////////////////权重，bias数据
    Convolution *player = dynamic_cast<Convolution *>(bench_conv->layer_);
    player->weight_data.create(c * kernel_size * kernel_size * out_channel);
    player->bias_data.create(out_channel);
    float *weight =(float*) player->weight_data.data;
    float *pweight=weight;
    float *bias = (float *)player->bias_data.data;
    for (int out = 0; out < out_channel; out++)
    {
        pweight = weight + c * kernel_size * kernel_size * out;
        for (int p = 0; p < c; p++)
        {
            for (int i = 0; i < kernel_size * kernel_size; i++)
            {
                pweight[i]=1.0f;
                //pweight[i] = rand() / (float)(RAND_MAX);
            }
            pweight += kernel_size * kernel_size;
        }
        bias[out] = 1;//rand() / (float)(RAND_MAX);
    }
    /////////////////////////////////////////////进行benchmark
    Mat out_mat;
    bench_conv->benchmark_layer(in_mat, out_mat);
    bench_conv->print_benchmark();
    int out_w=out_mat.w;
    int out_h=out_mat.h;
    int out_c=out_mat.c;
    if (!benchmark)
    {
        for(int c=0;c<out_c;c++)
        {
            float *data = (float *)out_mat.channel(c);
            for(int i=0;i<out_h;i++)
            {
                for(int j=0;j<out_h;j++)
                {
                    printf("%f ", *data);
                    data++;
                }
                printf("\n");
            }
            printf("\n");
        }
    }
        ///////////////////////////////////////////释放内存
    // if (NULL != weight)
    // {
    //     delete[] weight;
    //     weight = NULL;
    // }
    // if (NULL != bias)
    // {
    //     delete[] bias;
    //     bias = NULL;
    // }
}