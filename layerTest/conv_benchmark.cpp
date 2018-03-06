#include <iostream>
#include <stdio.h>
#include <vector>
#include <sys/time.h>

#include "layer.h"
#include "layer/arm/convolution_arm.h"
#include "layer/arm/deconvolution_arm.h"
#include "layer/deconvolution.h"
#include "paramdict.h"

using namespace ncnn;
using namespace std;

class BenchLayer
{
    public:
    BenchLayer():layer_(NULL),circle_num_(0),time_all_(0.0f),time_average_(0.0f){}
    BenchLayer(int number);
    virtual ~BenchLayer(){};

    double get_current_time()
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);

        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    }

    virtual int benchmark_layer(Mat & bottom_blob, Mat & top_blob);
    virtual int benchmark_layer(vector<Mat> &bottom_blobs, vector<Mat> &top_blobs);
    virtual int print_benchmark();

    public:
    Layer * layer_;
    ParamDict * paramdict_;
    int circle_num_;
    double time_all_;
    double time_average_;
};
/////////////////////////////////////////////////////////////////////////////
BenchLayer::BenchLayer(int number)
{

    //layer_ = new Convolution_arm;
    layer_ = new Deconvolution_arm;

    paramdict_ = new ParamDict;
    circle_num_ = number;
 }

 int BenchLayer::benchmark_layer(Mat &bottom_blob, Mat &top_blob)
 {
     if (layer_->support_inplace)
     {
         double start=get_current_time();
         for (int i = 0; i < circle_num_; i++)
         {
             layer_->forward_inplace(bottom_blob);
         }
         double end = get_current_time();
         time_all_ =end-start;
         time_average_=time_all_/circle_num_;
     }
     else
     {
         double start = get_current_time();
         for (int i = 0; i < circle_num_; i++)
         {
             Deconvolution_arm *player = dynamic_cast<Deconvolution_arm *>(layer_);
             player->forward(bottom_blob, top_blob);
         }
         double end = get_current_time();
         time_all_ = end - start;
         time_average_ = time_all_ / circle_num_;
     }
     return 0;
}

int BenchLayer::benchmark_layer(vector<Mat> &bottom_blobs, vector<Mat> &top_blobs)
{
    if (layer_->support_inplace)
    {
        double start = get_current_time();
        for (int i = 0; i < circle_num_; i++)
        {
            layer_->forward_inplace(bottom_blobs);
        }
        double end = get_current_time();
        time_all_ = end - start;
        time_average_ = time_all_ / circle_num_;
    }
    else
    {
        double start = get_current_time();
        for (int i = 0; i < circle_num_; i++)
        {
            layer_->forward(bottom_blobs, top_blobs);
        }
        double end = get_current_time();
        time_all_ = end - start;
        time_average_ = time_all_ / circle_num_;
    }
    return 0;
}

int BenchLayer::print_benchmark()
{
    cout<<"run "<<circle_num_<<" times used time "<<time_all_<<" ,average time is "<<time_average_<<endl;
}

int main(int argc,char** argv)
{
    cout<<"Benchmark_begin"<<endl;
    int circle_num=100;
    BenchLayer * bench_conv=new BenchLayer(circle_num);
    int kernel_size=4;
    int stride=2;
    int pad=1;
    int out_channel = 10;
    bench_conv->paramdict_->set(0, out_channel);
    bench_conv->paramdict_->set(1, kernel_size); //kernel size
    bench_conv->paramdict_->set(3, stride);
    //bench_conv->paramdict_->set(4, pad);         //pad
    bench_conv->paramdict_->set(5,1);            //bias enable
    bench_conv->layer_->load_param(*(bench_conv->paramdict_)); //配置层里面的参数
    ////////////////////////////////////////////////生成数据
    ////////////输入数据
    printf("Run to line %s %d\n",__FILE__,__LINE__);
    int w=500;
    int h=500;
    int c=10;
    Mat in_mat(w,h,c,sizeof(float));
    srand((unsigned)time(NULL));
    for(int p=0;p<c;p++)
    {
        float* data=in_mat.channel(p);
        for(int i=0;i<w*h;i++)
        {
            data[i] = rand()/(float)(RAND_MAX);
        }
    }
    printf("Run to line %s %d\n", __FILE__, __LINE__);
    /////////////////////权重，bias数据
    float *weight = new float[c * kernel_size * kernel_size * out_channel];
    float *bias=new float[c*out_channel];
    float *pweight=weight;
    printf("Run to line %s %d\n", __FILE__, __LINE__);
    for(int out=0;out<out_channel;out++)
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
    printf("Run to line %s %d\n", __FILE__, __LINE__);
    ///////////////////////////////////////////////配置层的weight等
    Deconvolution *player = dynamic_cast<Deconvolution *>(bench_conv->layer_);
    player->weight_data.data = weight;
    player->bias_data.data = bias;
    printf("Run to line %s %d\n", __FILE__, __LINE__);
    /////////////////////////////////////////////进行benchmark
    Mat out_mat;
    bench_conv->benchmark_layer(in_mat, out_mat);
    bench_conv->print_benchmark();
    printf("Run to line %s %d\n", __FILE__, __LINE__);
    ///////////////////////////////////////////释放内存
    if (NULL != weight)
    {
        delete [] weight;
        weight=NULL;
    }
    if (NULL != bias)
    {
        delete[] bias;
        bias = NULL;
    }
}