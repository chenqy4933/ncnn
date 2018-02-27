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

#include "benchmark.h"

#if NCNN_BENCHMARK
#include <stdio.h>
#include "layer/convolution.h"
#endif // NCNN_BENCHMARK

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

namespace ncnn {

#ifdef _WIN32
double get_current_time()
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    struct timeval tp;

    tp.tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp.tv_usec = (long) (system_time.wMilliseconds * 1000);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#else // _WIN32
double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif // _WIN32

#if NCNN_BENCHMARK

void benchmark(const Layer* layer, double start, double end)
{
    fprintf(stderr, "%-24s %-24s %8.2lfms", layer->type.c_str(), layer->name.c_str(), end - start);
    fprintf(stderr, "    |");
    fprintf(stderr, "\n");
}

void benchmark(const Layer* layer, const Mat& bottom_blob, Mat& top_blob, double start, double end)
{
    fprintf(stderr, "%-24s %-24s %8.2lfms", layer->type.c_str(), layer->name.c_str(), end - start);
    fprintf(stderr, "    |    feature_map: %4d x %-4d    inch: %4d    outch: %4d", bottom_blob.w, bottom_blob.h, bottom_blob.c, top_blob.c);
    if (layer->type == "Convolution")
    {
        fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d",
                ((Convolution*)layer)->kernel_w,
                ((Convolution*)layer)->kernel_h,
                ((Convolution*)layer)->stride_w,
                ((Convolution*)layer)->stride_h
        );
    }
    fprintf(stderr, "\n");
    // const char * name=layer->name.c_str();
    // FILE * pfile=fopen(name,"w+");
    // int c = top_blob.c;
    // int h = top_blob.h;
    // int w = top_blob.w;
    // float *data = (float *)top_blob.data;
    // for (int i = 0; i < c; i++)
    // {
    //     for(int j=0;j<h;j++)
    //     {
    //         for(int k=0;k<w;k++)
    //         {
    //             fprintf(pfile, "%60.55f ", *data);
    //             data++;
    //         }
    //         fprintf(pfile, "\n");
    //     }
    //     fprintf(pfile, "\n");
    // }
}

void benchmark(const Layer *layer, const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, double start, double end)
{
    fprintf(stderr, "%-24s %-24s %8.2lfms", layer->type.c_str(), layer->name.c_str(), end-start);
    fprintf(stderr, "    |");
    fprintf(stderr, "\n");

    // const char *name = layer->name.c_str();
    // FILE *pfile = fopen(name, "w+");

    // int size_blob = top_blobs.size();
    // for (int index = 0; index < size_blob; index++)
    // {
    //     float *data = (float *)top_blobs[index].data;
    //     int c = top_blobs[index].c;
    //     int h = top_blobs[index].h;
    //     int w = top_blobs[index].w;
    //     for (int i = 0; i < c; i++)
    //     {
    //         for (int j = 0; j < h; j++)
    //         {
    //             for (int k = 0; k < w; k++)
    //             {
    //                 fprintf(pfile, "%60.55f ", *data);
    //                 data++;
    //             }
    //             fprintf(pfile, "\n");
    //         }
    //         fprintf(pfile, "\n");
    //     }
    // }
    // fclose(pfile);
}

#endif // NCNN_BENCHMARK

} // namespace ncnn
