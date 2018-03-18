#include <iostream>
#include <stdio.h>
#include "benchlayer.h"
#include "layer/arm/batchnorm_arm.h"
#include "layer/batchnorm.h"
#define TEST_ACCURACY
using namespace std;

int test_benchmark(void)
{
    cout<<"Benchmark_begin"<<endl;
#ifdef TEST_ACCURACY
    bool benchmark=false;
#else
    bool benchmark=true;
#endif
    ////////////////////////////////////////////生成对用的层
    //const char* layer_name="PReLU" ;  //在不同的平台下面生成对应的类，如在arm平台下面会生成Convolution_arm,在PC下面会生成Convolution_x86
    const char* layer_name="BatchNorm" ;
#ifdef TEST_ACCURACY
    int circle_num=1;
#else
    int circle_num=1000;
#endif
    BenchLayer *bench_BatchNorm = new BenchLayer(layer_name,circle_num);
    ///////////////////////////////////////////配置层的参数，根据各层里面定义的参数顺序来定义
#ifdef TEST_ACCURACY
    int w=10;
    int h=10;
    int c=1;
#else
    int w=800;
    int h=800;
    int c=4;
#endif
    int num_slope=1;
    float slope=0.5f;
    bench_BatchNorm->paramdict_->set(0, c);
    //bench_prelu->paramdict_->set(0, slope);
    bench_BatchNorm->layer_->load_param(*(bench_BatchNorm->paramdict_)); //配置层里面的参数
    ////////////////////////////////////////////////生成数据
    ////////////输入数据
    
    Mat in_mat(w,h,c,sizeof(float));
    srand((unsigned)time(NULL));
    for(int p=0;p<c;p++)
    {
        float* data=in_mat.channel(p);
        for(int i=0;i<w*h;i++)
        {
#ifdef TEST_ACCURACY
            data[i]=((float)i)*0.1f;
#else
            data[i] = rand()/(float)(RAND_MAX);
#endif
        }
    }
    /////////////////////权重，bias数据
    BatchNorm *player = dynamic_cast<BatchNorm *>(bench_BatchNorm->layer_);
    player->a_data.create(c);
    player->b_data.create(c);
    float *aptr =(float*) player->a_data.data;
    float *pa=aptr;
    float *bptr =(float*) player->b_data.data;
    float *pb=bptr;
    for (int i = 0; i < c; i++)
    {
#ifdef TEST_ACCURACY
        *pa=1.0f;
        *pb=2.0f;
        pa++;
        pb++;
#else
        pa[i] = rand() / (float)(RAND_MAX);
        pb[i] = rand() / (float)(RAND_MAX);
#endif
    }
    /////////////////////////////////////////////进行benchmark
    Mat out_mat;
    bench_BatchNorm->benchmark_layer(in_mat, out_mat);
    bench_BatchNorm->print_benchmark();
    int out_w=in_mat.w;
    int out_h=in_mat.h;
    int out_c=in_mat.c;
    if (!benchmark)
    {
        for(int c=0;c<out_c;c++)
        {
            float *data = (float *)in_mat.channel(c);
            for(int i=0;i<out_h;i++)
            {
                for(int j=0;j<out_w;j++)
                {
                    printf("%1.3f ", *data);
                    data++;
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    return 0;
}


