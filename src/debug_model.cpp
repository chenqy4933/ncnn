

#include "debug_model.h"

#if NCNN_DEBUG_FILE
#include<unistd.h>
#include <sys/stat.h>
#include <pthread.h>


#define CAFFE_OUTPUT_BLOB_BIN_BASE_PATH "/Users/meitu/Desktop/code2"

void GetInputBlobBinPath(char* path)
{
  sprintf(path, CAFFE_OUTPUT_BLOB_BIN_BASE_PATH "/input/%lu", (unsigned long)pthread_self());
}

void GetOutputBlobBinPath(char* path)
{
  sprintf(path, CAFFE_OUTPUT_BLOB_BIN_BASE_PATH "/output/%lu", (unsigned long)pthread_self());
}

namespace ncnn {

Debug_file::Debug_file()
{
}

Debug_file::~Debug_file()
{
}


int Debug_file::df_mkdir()
{   
    //run_id = GetRunId();
    GetInputBlobBinPath(input_path);
    GetOutputBlobBinPath(output_path);
    if (run_id == 0)
    {
      mkdir(input_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      mkdir(output_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    run_id++;
    char dir_path[100];
    sprintf(dir_path, "%s/%d", input_path, run_id);
    mkdir(dir_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    sprintf(dir_path, "%s/%d", output_path, run_id);
    mkdir(dir_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    return 0;
}

int Debug_file::write_inputs(std::vector<Mat> &blobs,int layer_index) const
{
    char const *path = (char const *)input_path;
    df_writes(blobs,layer_index,path);
    return 0;
}
int Debug_file::write_outputs(std::vector<Mat> &blobs,int layer_index) const
{
    char const *path = output_path;
    df_writes(blobs,layer_index,path);
    return 0;
}

int Debug_file::write_input(Mat &blob,int layer_index) const
{
    char const *path = input_path;
    df_write(blob,layer_index,path);
    return 0;
}
int Debug_file::write_output(Mat &blob,int layer_index) const
{
    char const *path = output_path;
    df_write(blob,layer_index,path);
    return 0;
}

}

#endif

