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

#include "prelu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(PReLU_arm)

// int PReLU_arm::forward_inplace(Mat& bottom_top_blob) const
// {
//     int dims = bottom_top_blob.dims;
//     if (dims != 3)
//         return PReLU::forward_inplace(bottom_top_blob);

//     int w = bottom_top_blob.w;
//     int h = bottom_top_blob.h;
//     int channels = bottom_top_blob.c;
//     int size = w * h;

//     const float* slope_data_ptr = slope_data;

//     #pragma omp parallel for
//     for (int q=0; q<channels; q++)
//     {
//         float* ptr = bottom_top_blob.channel(q);
//         float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

// #if __ARM_NEON
//         int nn = size >> 2;
//         int remain = size - (nn << 2);
// #else
//         int remain = size;
// #endif // __ARM_NEON

// #if __ARM_NEON
// #if __aarch64__
//         // float32x4_t _zero = vdupq_n_f32(0.f);
//         // float32x4_t _slope = vdupq_n_f32(slope);
//         // for (; nn>0; nn--)
//         // {
//         //     float32x4_t _p = vld1q_f32(ptr);
//         //     uint32x4_t _lemask = vcleq_f32(_p, _zero);
//         //     float32x4_t _ps = vmulq_f32(_p, _slope);
//         //     _p = vbslq_f32(_lemask, _ps, _p);
//         //     vst1q_f32(ptr, _p);

//         //     ptr += 4;
//         // }
//         if (nn > 0)
//         {
//             float32x4_t _slope = vdupq_n_f32(slope);
//             asm volatile(
//                 "0:                                        \n"
//                 "prfm   pldl1keep, [%1, #128]              \n"
//                 "ld1    {v0.4s}, [%1]                      \n"

//                 "fcmle   v3.4s, v0.4s, #0.0                \n"
//                 "fmul     v4.4s, v0.4s, %4.4s               \n"
//                 "bit     v0.16b, v4.16b, v3.16b             \n"
//                 "subs     %0 ,%0, #1                        \n"
//                 "st1     {v0.4s}, [%1],#16                 \n"
//                 "bne        0b                             \n"
//                 : "=r"(nn), // %0
//                   "=r"(ptr) // %1
//                 : "0"(nn),
//                   "1"(ptr),
//                   "w"(_slope) // %4
//                 : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
//         }
// #else
//         if (nn > 0)
//         {
//         asm volatile(
//             "veor       q1, q0, q0          \n"
//             "vdup.f32   q2, %4              \n"
//             "0:                             \n"
//             "pld        [%1, #128]          \n"
//             "vld1.f32   {d0-d1}, [%1 :128]  \n"
//             "vcle.f32   q3, q0, q1          \n"
//             "vmul.f32   q4, q0, q2          \n"
//             "vbit.32    q0, q4, q3          \n"
//             "subs       %0, #1              \n"
//             "vst1.f32   {d0-d1}, [%1 :128]! \n"
//             "bne        0b                  \n"
//             : "=r"(nn),     // %0
//               "=r"(ptr)     // %1
//             : "0"(nn),
//               "1"(ptr),
//               "r"(slope)    // %4
//             : "cc", "memory", "q0", "q1", "q2", "q3", "q4"
//         );
//         }
        
//         // if (nn > 0)
//         // {
//         //     asm volatile(
//         //         "veor       q1, q0, q0          \n"
//         //         "vdup.f32   q2, %4              \n"
//         //         "0:                             \n"
//         //         "pld        [%1, #128]          \n"
//         //         "vld1.f32   {d0-d1}, [%1 :128]  \n"
//         //         "vmax.f32   q3, q0, q1          \n"
//         //         "vmin.f32   q4, q0, q1          \n"
//         //         "vmla.f32   q3, q4, q2          \n"

//         //         "subs       %0, #1              \n"
//         //         "vst1.f32   {d6-d7}, [%1 :128]! \n"
//         //         "bne        0b                  \n"
//         //         : "=r"(nn), // %0
//         //           "=r"(ptr) // %1
//         //         : "0"(nn),
//         //           "1"(ptr),
//         //           "r"(slope) // %4
//         //         : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
//         // }
// #endif // __aarch64__
// #endif // __ARM_NEON
//         for (; remain>0; remain--)
//         {
//             if (*ptr < 0)
//                 *ptr *= slope;

//             ptr++;
//         }
//     }

//     return 0;
// }

int PReLU_arm::forward_inplace(Mat &bottom_top_blob) const
{
    int dims = bottom_top_blob.dims;
    if (dims != 3)
        return PReLU::forward_inplace(bottom_top_blob);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    const float *slope_data_ptr = slope_data;

#pragma omp parallel for
    for (int q = 0; q < channels; q++)
    {
        float *ptr = bottom_top_blob.channel(q);
        float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

#if __ARM_NEON
        int nn = size >> 4;
        int remain = size - (nn << 4);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        if (nn > 0)
        {
            float32x4_t _slope = vdupq_n_f32(slope);
            asm volatile(

                "eor   v1.16b,  v0.16b,  v0.16b              \n"

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
                "fmla   v2.4s, v3.4s, %4.4s                \n"
                "st1     {v2.4s}, [%1],#16                 \n"

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
                "fmla   v2.4s, v3.4s, %4.4s                \n"
                "st1     {v2.4s}, [%1],#16                 \n"

                "subs     %0 ,%0, #1                        \n"
                "bne        0b                             \n"
                : "=r"(nn), // %0
                  "=r"(ptr) // %1
                : "0"(nn),
                  "1"(ptr),
                  "w"(_slope) // %4
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
        }
#else
        if (nn > 0)
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

                "subs    %0, #1              \n"
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
        remain = remain - (remain_nn << 2);
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
        if (remain_nn > 0)
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
        for (; remain > 0; remain--)
        {
            if (*ptr < 0)
                *ptr *= slope;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
