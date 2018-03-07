
#ifndef NCNN_EIGEN_H_
#define NCNN_EIGEN_H_

#include <Eigen/Dense>
#include <assert.h>
#include "im2col.h"

enum  	CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum  	CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
#define MAP_SVECTOR2(name, ptr, N, step) Eigen::Map<Eigen::VectorXf, 0, Eigen::InnerStride<> > name(ptr, N, Eigen::InnerStride<>(step))
#define MAP_CONST_SVECTOR2(name, ptr, N, step) Eigen::Map<const Eigen::VectorXf, 0, Eigen::InnerStride<> > name(ptr, N, Eigen::InnerStride<>(step))
#define MAP_DVECTOR2(name, ptr, N, step) Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<> > name(ptr, N, Eigen::InnerStride<>(step))
#define MAP_CONST_DVECTOR2(name, ptr, N, step) Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<> > name(ptr, N, Eigen::InnerStride<>(step))
#define MAP_SVECTOR(name, ptr, N) Eigen::Map<Eigen::VectorXf> name(ptr, N)
#define MAP_CONST_SVECTOR(name, ptr, N) Eigen::Map<const Eigen::VectorXf> name(ptr, N)
#define MAP_DVECTOR(name, ptr, N) Eigen::Map<Eigen::VectorXd> name(ptr, N)
#define MAP_CONST_DVECTOR(name, ptr, N) Eigen::Map<const Eigen::VectorXd> name(ptr, N)
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXf;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXd;
#define MAP_SMATRIX(name, ptr, M, N) Eigen::Map<MatXf> name(ptr, M, N)
#define MAP_CONST_SMATRIX(name, ptr, M, N) Eigen::Map<const MatXf> name(ptr, M, N)
#define MAP_DMATRIX(name, ptr, M, N) Eigen::Map<MatXd> name(ptr, M, N)
#define MAP_CONST_DMATRIX(name, ptr, M, N) Eigen::Map<const MatXd> name(ptr, M, N)



inline void conv_col2im_cpu(const float* col_buff, float* data, const ncnn::Mat& bottomBlob,
	ncnn::Mat& topBlob, int pad_h, int pad_w, int kernel, int stride) 
{
	ncnn::col2im_cpu(col_buff, topBlob.c,
	  topBlob.h, topBlob.w,
	  topBlob.cstep, bottomBlob.cstep,
	  kernel, kernel,
	  pad_h, pad_w,
	  stride, stride,
	  1, 1, data);
}

inline void conv_col2im_cpu(const float* col_buff, float* data, 
	const ncnn::Mat& bottomBlob, ncnn::Mat& topBlob, 
	int pad_h, int pad_w, int kernel_h, int kernel_w, 
	int stride_h, int stride_w, int dilation_h, int dilation_w) 
{
	ncnn::col2im_cpu(col_buff, topBlob.c,
	  topBlob.h, topBlob.w,
	  topBlob.cstep, bottomBlob.cstep,
	  kernel_h, kernel_w,
	  pad_h, pad_w,
	  stride_h, stride_w,
	  dilation_h, dilation_w, data);
}


static int deconv_eigen(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, 
	const ncnn::Mat& _kernel, const ncnn::Mat& _bias, 
	int pad_h, int pad_w, 
	int kernel_h, int kernel_w,
	int stride_h, int stride_w,
	int dilation_h,int dilation_w
	)
{
	//M  	: 	kernel_dim_ 					weight权值的大小(num_*kh*kw)
	//N		:	conv_out_spatial_dim_			bottom_blob的大小(h*w)		
	//beta	:	0
	//K		:	conv_out_channels_ / group_		bottom_blob的通道(c/g)
	//alpha	:	1
	//A		:	weights + weight_offset_ * g
	//B		:	output + output_offset_ * g

	//TODO: bottom align data should be set as 0.

	//allocate weight to exchange num and channel.
	//TODO: no alloc.
	ncnn::Mat _kernelchange(kernel_h, kernel_w, bottom_blob.c, top_blob.c, _kernel);
	
	int M 		= top_blob.c * kernel_h * kernel_w;
	int N 		= bottom_blob.cstep;
	float beta 	= 0.;
	int K		= bottom_blob.c;
	float alpha	= 1.;
	float* A	= (float *)_kernelchange.data;
	float* B	= (float *)bottom_blob.data;

	const float* bias = (const float*)_bias.data;

	for (int p=0; p<top_blob.c; p++)
    {
		ncnn::Mat out = top_blob.channel(p);
		const float bias0 = bias ? bias[p] : 0.f;
	    out.fill(bias0);
	}
	
	float* outv	= (float *)top_blob.data;
	int CM 	= N * M;
	float* C	= (float *)malloc(CM * sizeof(float));
	assert(C);
		
	MAP_SMATRIX(eC, C, M, N);
    eC *= beta;
	MAP_CONST_SMATRIX(eA, A, K, M);
	MAP_CONST_SMATRIX(eB, B, K, N);
	eC.noalias() += alpha * (eA.transpose() * eB);

	conv_col2im_cpu(C, outv, bottom_blob, top_blob, pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w);

	free(C);

	return 1;
}

static int deconv4x4s2_eigen(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Mat& _kernel, const ncnn::Mat& _bias, int pad_h, int pad_w)
{
	return deconv_eigen(bottom_blob,top_blob,_kernel,_bias,pad_h,pad_w,4,4,2,2,1,1);
}

#endif
