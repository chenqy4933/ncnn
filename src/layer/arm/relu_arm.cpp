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

#include "relu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(ReLU_arm)

int ReLU_arm::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

#if 0 // 同步caffemobile的relu版本, 实现方式能获得更高性能
    int align_size = 16;
    int nn = size & -align_size;
	#pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* pBottom_data = (float*)bottom_top_blob.channel(q);
        
        float32x4_t zero_4 = vdupq_n_f32(0.0f);
        float32x4_t slope_4 = vdupq_n_f32(slope);
        int i;
        //通用版本
        for(i = 0; i < nn; i += align_size)
        {
            float32x4_t data1 = vld1q_f32(pBottom_data + 0);
            float32x4_t data2 = vld1q_f32(pBottom_data + 4);
            float32x4_t data3 = vld1q_f32(pBottom_data + 8);
            float32x4_t data4 = vld1q_f32(pBottom_data + 12);
            
            float32x4_t data1_max = vmaxq_f32(data1, zero_4);
            float32x4_t data2_max = vmaxq_f32(data2, zero_4);
            float32x4_t data3_max = vmaxq_f32(data3, zero_4);
            float32x4_t data4_max = vmaxq_f32(data4, zero_4);
            
            float32x4_t data1_min = vminq_f32(data1, zero_4);
            float32x4_t data2_min = vminq_f32(data2, zero_4);
            float32x4_t data3_min = vminq_f32(data3, zero_4);
            float32x4_t data4_min = vminq_f32(data4, zero_4);
            
            float32x4_t result1 = vmlaq_f32(data1_max, slope_4, data1_min);
            float32x4_t result2 = vmlaq_f32(data2_max, slope_4, data2_min);
            float32x4_t result3 = vmlaq_f32(data3_max, slope_4, data3_min);
            float32x4_t result4 = vmlaq_f32(data4_max, slope_4, data4_min);
            
            vst1q_f32(pBottom_data + 0, result1);
            vst1q_f32(pBottom_data + 4, result2);
            vst1q_f32(pBottom_data + 8, result3);
            vst1q_f32(pBottom_data + 12, result4);
            
            pBottom_data += align_size;
        }
        for(; i < size; i++)
        {
            float tmp_data = *pBottom_data;
            *pBottom_data = tmp_data > float(0) ? tmp_data : tmp_data *  slope;
            pBottom_data++;
        }
    }
    
    return 0;
#else
    if (slope == 0.f)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 4;
            int remain = size - (nn << 4);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            // float32x4_t _zero = vdupq_n_f32(0.f);
            // for (; nn>0; nn--)
            // {
            //     float32x4_t _p = vld1q_f32(ptr);
            //     _p = vmaxq_f32(_p, _zero);
            //     vst1q_f32(ptr, _p);

            //     ptr += 4;
            // }
            
            if (nn > 0)
            {
                asm volatile(

                    "eor    v1.16b, v0.16b, v0.16b          \n"

                    "0:                                        \n"
                    "prfm   pldl1keep, [%1, #128]              \n"
                    "ld1    {v0.4s}, [%1]                      \n"
                    "fmax   v0.4s, v0.4s, v1.4s                \n"
                    "st1     {v0.4s}, [%1],#16                 \n"

                    "prfm   pldl1keep, [%1, #128]              \n"
                    "ld1    {v0.4s}, [%1]                      \n"
                    "fmax   v0.4s, v0.4s, v1.4s                \n"
                    "st1     {v0.4s}, [%1],#16                 \n"

                    "prfm   pldl1keep, [%1, #128]              \n"
                    "ld1    {v0.4s}, [%1]                      \n"
                    "fmax   v0.4s, v0.4s, v1.4s                \n"
                    "st1     {v0.4s}, [%1],#16                 \n"

                    "prfm   pldl1keep, [%1, #128]              \n"
                    "ld1    {v0.4s}, [%1]                      \n"
                    "fmax   v0.4s, v0.4s, v1.4s                \n"
                    "st1     {v0.4s}, [%1],#16                 \n"
                  
                    "subs     %0,  %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn), // %0
                      "=r"(ptr) // %1
                    : "0"(nn),
                      "1"(ptr)
                    : "cc", "memory", "v0", "v1");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "veor       q1, q0, q0          \n"
                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "vst1.f32   {d0-d1}, [%1 :128]! \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "vst1.f32   {d0-d1}, [%1 :128]! \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "vst1.f32   {d0-d1}, [%1 :128]! \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "vst1.f32   {d0-d1}, [%1 :128]! \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn), // %0
                      "=r"(ptr) // %1
                    : "0"(nn),
                      "1"(ptr)
                    : "cc", "memory", "q0", "q1");
            }
#endif // __aarch64__
            int remain_nn=remain>>2;
            remain=(remain-remain_nn<<2);
#if __aarch64__
            if (remain_nn>0)
            {
                asm volatile(

                    "eor       v1.16b, v0.16b, v0.16b          \n"

                    "0:                                        \n"
                    "prfm   pldl1keep, [%1, #128]              \n"
                    "ld1    {v0.4s}, [%1]                      \n"
                    "fmax   v0.4s, v0.4s, v1.4s                \n"
                    "st1     {v0.4s}, [%1],#16                 \n"

                    "subs     %0, %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(remain_nn), // %0
                      "=r"(ptr)        // %1
                    : "0"(remain_nn),
                      "1"(ptr)
                    : "cc", "memory", "v0", "v1");
            }
#else
            if(remain_nn>0)
            {
                asm volatile(
                    "veor       q1, q0, q0          \n"
                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "vst1.f32   {d0-d1}, [%1 :128]! \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(remain_nn), // %0
                      "=r"(ptr)        // %1
                    : "0"(remain_nn),
                      "1"(ptr)
                    : "cc", "memory", "q0", "q1");
            }
#endif
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *ptr = std::max(*ptr, 0.f);

                ptr++;
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 4;
            int remain = size - (nn << 4);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            // float32x4_t _zero = vdupq_n_f32(0.f);
            // float32x4_t _slope = vdupq_n_f32(slope);
            // for (; nn>0; nn--)
            // {
            //     float32x4_t _p = vld1q_f32(ptr);
            //     uint32x4_t _lemask = vcleq_f32(_p, _zero);
            //     float32x4_t _ps = vmulq_f32(_p, _slope);
            //     _p = vbslq_f32(_lemask, _ps, _p);
            //     vst1q_f32(ptr, _p);

            //     ptr += 4;
            // }
            if(nn>0)
            {
                float32x4_t _slope = vdupq_n_f32(slope);
                asm volatile(

                    "eor       v1.16b, v0.16b, v0.16b          \n"

                    "0:                                        \n"
                    "prfm   pldl1keep, [%1, #128]              \n"
                    "ld1    {v0.4s}, [%1]                      \n"
                    "fmax   v2.4s, v0.4s, v1.4s                \n"
                    "fmin   v3.4s, v0.4s, v1.4s               \n"
                    "fmla   v2.4s, v3.4s, %4.4s                \n"
                    "st1     {v2.4s}, [%1],#16                 \n"

                    "prfm   pldl1keep, [%1, #128]              \n"
                    "ld1    {v0.4s}, [%1]                      \n"
                    "fmax   v2.4s, v0.4s, v1.4s                \n"
                    "fmin   v3.4s, v0.4s, v1.4s               \n"
                    "fmla   v2.4s, v3.4s, %4.4s             \n"
                    "st1     {v2.4s}, [%1],#16                 \n"

                    "prfm   pldl1keep, [%1, #128]              \n"
                    "ld1    {v0.4s}, [%1]                      \n"
                    "fmax   v2.4s, v0.4s, v1.4s                \n"
                    "fmin   v3.4s, v0.4s, v1.4s               \n"
                    "fmla   v2.4s, v3.4s, %4.4s             \n"
                    "st1     {v2.4s}, [%1],#16                 \n"

                    "prfm   pldl1keep, [%1, #128]              \n"
                    "ld1    {v0.4s}, [%1]                      \n"
                    "fmax   v2.4s, v0.4s, v1.4s                \n"
                    "fmin   v3.4s, v0.4s, v1.4s               \n"
                    "fmla   v2.4s, v3.4s, %4.4s             \n"
                    "st1     {v2.4s}, [%1],#16                 \n"

                    "subs     %0 ,%0, #1                        \n"
                    "bne        0b                             \n"
                    : "=r"(nn), // %0
                    "=r"(ptr) // %1
                    : "0"(nn),
                    "1"(ptr),
                    "w"(_slope) // %4
                    : "cc", "memory", "v0", "v1", "v2", "v3");
            }
#else
            if(nn>0)
            {
                asm volatile(
                    "veor       q1, q0, q0          \n"
                    "vdup.f32   q2, %4              \n"

                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q3, q0, q1          \n"
                    "vmin.f32   q4, q0, q1          \n"
                    "vmla.f32   q3, q4, q2          \n"
                    "vst1.f32   {d6-d7}, [%1 :128]! \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q3, q0, q1          \n"
                    "vmin.f32   q4, q0, q1          \n"
                    "vmla.f32   q3, q4, q2          \n"
                    "vst1.f32   {d6-d7}, [%1 :128]! \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q3, q0, q1          \n"
                    "vmin.f32   q4, q0, q1          \n"
                    "vmla.f32   q3, q4, q2          \n"
                    "vst1.f32   {d6-d7}, [%1 :128]! \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q3, q0, q1          \n"
                    "vmin.f32   q4, q0, q1          \n"
                    "vmla.f32   q3, q4, q2          \n"
                    "vst1.f32   {d6-d7}, [%1 :128]! \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn), // %0
                    "=r"(ptr) // %1
                    : "0"(nn),
                    "1"(ptr),
                    "r"(slope) // %4
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
            }
#endif // __aarch64__
        int remain_nn = remain >> 2;
        remain=remain-remain_nn<<2;
#if __aarch64__
        if (remain_nn)
        {
            float32x4_t _slope = vdupq_n_f32(slope);
            asm volatile(

                "eor       v1.16b, v0.16b, v0.16b          \n"

                "0:                                        \n"
                "prfm   pldl1keep, [%1, #128]              \n"
                "ld1    {v0.4s}, [%1]                      \n"
                "fmax   v2.4s, v0.4s, v1.4s                \n"
                "fmin   v3.4s, v0.4s, v1.4s               \n"
                "fmla   v2.4s, v3.4s, %4.4s             \n"
                "st1     {v2.4s}, [%1],#16                 \n"

                "subs     %0 ,%0, #1                        \n"
                "bne        0b                             \n"
                : "=r"(remain_nn), // %0
                  "=r"(ptr)        // %1
                : "0"(remain_nn),
                  "1"(ptr),
                  "w"(_slope) // %4
                : "cc", "memory", "v0", "v1", "v2", "v3");
        }
#else
        if (remain_nn>0)
        {
            asm volatile(
                "veor       q1, q0, q0          \n"
                "vdup.f32   q2, %4              \n"

                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vmax.f32   q3, q0, q1          \n"
                "vmin.f32   q4, q0, q1          \n"
                "vmla.f32   q3, q4, q2          \n"
                "vst1.f32   {d6-d7}, [%1 :128]! \n"

                "subs       %0, #1              \n"
                "bne        0b                  \n"
                : "=r"(remain_nn), // %0
                  "=r"(ptr)        // %1
                : "0"(remain_nn),
                  "1"(ptr),
                  "r"(slope) // %4
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8");
        }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                if (*ptr < 0)
                    *ptr *= slope;

                ptr++;
            }
        }
    }

    return 0;
#endif
}

} // namespace ncnn
