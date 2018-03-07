#include <iostream>
#include <stdio.h>
#include <vector>
#include <sys/time.h>

#include "layer.h"
#include "paramdict.h"

using namespace ncnn;
using namespace std;
  /***************************************************************************
  *  该类供ncnn内部层优化前后的自身对比，通过传入层的类型名字，创建对应的层。进行benchmark
  *
  ***************************************************************************/
class BenchLayer
{
  public:
    BenchLayer() : layer_(NULL), circle_num_(0), time_all_(0.0f), time_average_(0.0f) {}
    BenchLayer(const char *name, int number);
    virtual ~BenchLayer(){};

    double get_current_time()
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);

        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    }

    virtual int benchmark_layer(Mat &bottom_blob, Mat &top_blob);
    virtual int benchmark_layer(vector<Mat> &bottom_blobs, vector<Mat> &top_blobs);
    virtual void print_benchmark();

  public:
    Layer *layer_;
    ParamDict *paramdict_;
    const char * layer_name_;
    int circle_num_;
    double time_all_;
    double time_average_;
};
/////////////////////////////////////////////////////////////////////////////
BenchLayer::BenchLayer(const char *name, int number)
{

    //layer_ = new Convolution_arm;
    //layer_ = new Deconvolution_arm;
    layer_name_ = name;
    layer_ = create_layer(name);
    paramdict_ = new ParamDict;
    circle_num_ = number;
}

int BenchLayer::benchmark_layer(Mat &bottom_blob, Mat &top_blob)
{
    if (layer_->support_inplace)
    {
        double start = get_current_time();
        for (int i = 0; i < circle_num_; i++)
        {
            layer_->forward_inplace(bottom_blob);
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
            layer_->forward(bottom_blob, top_blob);
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

void BenchLayer::print_benchmark(void)
{
    cout <<layer_name_<< " run " << circle_num_ << " times used time " << time_all_ << " ,average time is " << time_average_ << endl;
    cout << "ParamDict 0:" << paramdict_->get(0, 0) << " 1:" << paramdict_->get(1, 0) << " 2:" << paramdict_->get(2, 0)
         << " 3:" << paramdict_->get(3, 0) << " 4:" << paramdict_->get(4, 0) << " 5:" << paramdict_->get(5, 0) << endl;
    //打印每一个参数对的时候，默认都设置为0。
}