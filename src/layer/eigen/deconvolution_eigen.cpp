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

#include "deconvolution_eigen.h"
#include "eigen.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Deconvolution_eigen)

int Deconvolution_eigen::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    int outw = (w - 1) * stride_w + dilation_w * (kernel_w -1 ) + 1 - 2*pad_w;
    int outh = (h - 1) * stride_h + dilation_h * (kernel_h -1 ) + 1 - 2*pad_h ;

    Mat top_blob_bordered = top_blob;
    top_blob_bordered.create(outw, outh, num_output);
    if (top_blob_bordered.empty())
        return -100;

    deconv_eigen(bottom_blob, top_blob_bordered, weight_data, bias_data,
        pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w);

    top_blob = top_blob_bordered;

    return 0;
}

} // namespace ncnn
