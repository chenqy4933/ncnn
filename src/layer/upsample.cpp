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

#include "upsample.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Upsample)

/***************************************************************************
*   变量与位置之间的对应关系
*   scale<------->0
*   scale_w<----->1
*   scale_h<----->11
*   upsample_w<-->2
*   upsample_h<-->12
*   pad_out_w_<-->3
*   pad_out_h_<-->13
***************************************************************************/


Upsample::Upsample()
{
    one_blob_only = false;
    support_inplace = false;
}

int Upsample::load_param(const ParamDict& pd)
{
    scale = pd.get(0, 0);
    scale_w = pd.get(1, 0);
    scale_h = pd.get(11, 0);

    upsample_w = pd.get(2, 0);
    upsample_h = pd.get(12, 0);

    pad_out_w_ = pd.get(3, 0);
    pad_out_h_ = pd.get(13, 0);
    
    ////////////////////这儿需要检查参数
    return 0;
}

int Upsample::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    int bottom_size=bottom_blobs.size();

    int height=bottom_blobs[0].h;
    int width=bottom_blobs[0].w;
    int channels=bottom_blobs[0].c;

    int size=height*width;
    int dst_w=upsample_w;
    int dst_h=upsample_h;

    if(bottom_size<2)
    {
        return -5;
    }

    if(scale>0)
    {
        dst_w=scale * width - pad_out_w_;
        dst_h=scale * height - pad_out_h_;
    }
    else 
    {
        if(scale_w!=0 && scale_h!=0)
        {
            dst_w=scale_w * width - pad_out_w_;
            dst_h=scale_h * height - pad_out_h_;
        }
    }
    if(dst_w==0 || dst_h==0)
    {
        return -5;
    }

    Mat bottom_blob=bottom_blobs[0];
    Mat bottom_index=bottom_blobs[1];

    Mat& top_blob=top_blobs[0];
    top_blob.create(dst_w,dst_h,channels);
    if(top_blob.empty())
    {
        return -100;
    }

   #pragma omp parallel for
    for (int q = 0; q < channels; ++q)
    {
        const float *ptr = bottom_blob.channel(q);
        const float *ptr_index=bottom_index.channel(q);

        float *output_ptr = top_blob.channel(q);
        for (int index= 0; index < size; ++index)
        {
            int position=static_cast<int>(ptr_index[index]);
            output_ptr[position]=ptr[index];
        }
    }
    return 0;
}

} // namespace ncnn
