#include <iostream>
#include <stdio.h>
#include "benchlayer.h"
#include "layer/arm/eltwise_arm.h"
#include "layer/eltwise.h"
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
    const char* layer_name="Eltwise" ;
#ifdef TEST_ACCURACY
    int circle_num=1;
#else
    int circle_num=1000;
#endif
    BenchLayer *bench_eltwise = new BenchLayer(layer_name,circle_num);
    ///////////////////////////////////////////配置层的参数，根据各层里面定义的参数顺序来定义
#ifdef TEST_ACCURACY
    int w=10;
    int h=10;
    int c=3;
#else
    int w=800;
    int h=800;
    int c=4;
#endif
    int blobs=3;
    bench_eltwise->paramdict_->set(0, blobs);
    bench_eltwise->layer_->load_param(*(bench_eltwise->paramdict_)); //配置层里面的参数
    ////////////////////////////////////////////////生成数据
    ////////////输入数据
    
    std::vector<Mat>in_mat(blobs);  //(w,h,c,sizeof(float));
    srand((unsigned)time(NULL));
    for(int b=0;b<blobs;b++)
    {
        in_mat[b].create(w,h,c);
        for(int p=0;p<c;p++)
        {
            float* data=in_mat[b].channel(p);
            for(int i=0;i<w*h;i++)
            {
    #ifdef TEST_ACCURACY
                data[i]=((float)i)*0.1f;
    #else
                data[i] = rand()/(float)(RAND_MAX);
    #endif
            }
        }
    }
    /////////////////////权重，bias数据
    Eltwise *player = dynamic_cast<Eltwise *>(bench_eltwise->layer_);
    player->op_type=Eltwise::Operation_SUM;
    bool need_coffes=true;
    int type=Eltwise::Operation_SUM;
    if((type==player->op_type) && need_coffes)
    {
        player->coeffs.create(blobs);
        float *aptr =(float*) player->coeffs.data;
        float *pa=aptr;
        for (int i = 0; i <blobs; i++)
        {
    #ifdef TEST_ACCURACY
            *pa=2;
            pa++;
    #else
            pa[i] = rand() / (float)(RAND_MAX);
            pb[i] = rand() / (float)(RAND_MAX);
    #endif
        }
    }
    /////////////////////////////////////////////进行benchmark
    std::vector<Mat> out_mat(1);
    bench_eltwise->benchmark_layer(in_mat, out_mat);
    bench_eltwise->print_benchmark();
    int out_w=out_mat[0].w;
    int out_h=out_mat[0].h;
    int out_c=out_mat[0].c;
    if (!benchmark)
    {
        for(int c=0;c<out_c;c++)
        {
            float *data = (float *)out_mat[0].channel(c);
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



