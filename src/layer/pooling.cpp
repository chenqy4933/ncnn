// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pooling.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Pooling)

Pooling::Pooling()
{
    one_blob_only = false;
    support_inplace = false;
}

int Pooling::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_w = pd.get(3, 0);
    pad_h = pd.get(13, pad_w);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);

    return 0;
}

int Pooling::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;
    int channels = bottom_blobs[0].c;
    bool mult_output=(top_blobs.size()>1);

//     fprintf(stderr, "Pooling     input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling && !mult_output)
    {
        top_blobs[0].create(1, 1, channels);
        if (top_blobs[0].empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blobs[0].channel(q);
                float* outptr = top_blobs[0].channel(q);

                float max = ptr[0];
                for (int i=0; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                outptr[0] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blobs[0].channel(q);
                float* outptr = top_blobs[0].channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                outptr[0] = sum / size;
            }
        }

        return 0;
    }
////////padding the input mat
    Mat bottom_blob_bordered = bottom_blobs[0];
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blobs[0], bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_mode == 2) // tensorflow padding=SAME
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blobs[0], bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    int wtail = 0;
    int htail = 0;
    if (pad_mode == 0) // full padding
    {
        wtail = (w - kernel_w) % stride_w;
        htail = (h - kernel_h) % stride_h;
    }
    if (wtail != 0 || htail != 0)
    {
        int wtailpad = 0;
        int htailpad = 0;
        if (wtail != 0)
            wtailpad = kernel_w - wtail;
        if (htail != 0)
            htailpad = kernel_h - htail;

        Mat bottom_blob_bordered2;
        if (pooling_type == PoolMethod_MAX)
        {
            copy_make_border(bottom_blob_bordered, bottom_blob_bordered2, 0, htailpad, 0, wtailpad, BORDER_REPLICATE, 0.f);
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            copy_make_border(bottom_blob_bordered, bottom_blob_bordered2, 0, htailpad, 0, wtailpad, BORDER_CONSTANT, 0.f);
        }
        if (bottom_blob_bordered2.empty())
            return -100;

        bottom_blob_bordered = bottom_blob_bordered2;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        if (wtail != 0)
            outw += 1;
        if (htail != 0)
            outh += 1;
    }
////////////////////////padding input over!!!
    top_blobs[0].create(outw, outh, channels);
    if (top_blobs[0].empty())
        return -100;
    //if need multiple out_put go to here
    if(mult_output)
    {
    		top_blobs[1].create(outw,outh,channels);
    		if(top_blobs[1].empty())
    			return -100;
    }

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blobs[0].channel(q);
            float * out_index;
            int max_index=0;
            if(mult_output)
            {
            		out_index=top_blobs[1].channel(q);
            }

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                		max_index=0;
                		int max_start=w*(i*stride_h)+j*stride_w;
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        max = std::max(max, val);
                        if(max==val)
                        {
                        		max_index=max_start+space_ofs[k];
                        }
                    }

                    outptr[j] = max;
                    if(mult_output)
                    {
                    		out_index[j]=max_index;
                    }
                }

                outptr += outw;
                if(mult_output)
				{
                		out_index+=outw;
				}
            }
        }
    }
    /////////////  均值pooling 不需要index blob
    else if (pooling_type == PoolMethod_AVE)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blobs[0].channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float sum = 0;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        sum += val;
                    }

                    outptr[j] = sum / maxk;
                }

                outptr += outw;
            }

            // fix tail pad
            if (wtail != 0)
            {
                const float scale = (float)kernel_w / (kernel_w - wtail);

                outptr = top_blobs[0].channel(q);
                outptr += outw - 1;
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
            if (htail != 0)
            {
                const float scale = (float)kernel_h / (kernel_h - htail);

                outptr = top_blobs[0].channel(q).row(outh - 1);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
