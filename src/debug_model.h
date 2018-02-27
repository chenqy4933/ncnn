
#define CAFFE_OUTPUT_BLOB_BIN_FILE 1

#ifndef NCNN_DEBUG_MODEL_H
#define NCNN_DEBUG_MODEL_H



#if CAFFE_OUTPUT_BLOB_BIN_FILE

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

#include "mat.h"


namespace ncnn {

class Debug_file
{
public:

    Debug_file();
	~Debug_file();
	
	int df_mkdir();
	int write_inputs(std::vector<Mat> &bottom_blobs,int layer_index) const;
	int write_outputs(std::vector<Mat> &bottom_blobs,int layer_index) const;
	int write_input(Mat &bottom_blob,int layer_index) const;
	int write_output(Mat &bottom_blob,int layer_index) const;	
	
protected:
	
	inline void df_writes(std::vector<Mat> &blobs,int layer_index, char const *path) const
	{
		int size = blobs.size(); 
		for (int x = 0; x < size; x++)
		{
			int len = blobs[x].total();
			const void* raw_data = blobs[x].data;
			FILE * outfile;
			char fileName[100];
			sprintf(fileName, "%s/%d/%d-%d.bin", path, run_id, layer_index, x);
			outfile = fopen(fileName, "wb" );
			if (outfile == NULL)
			  printf("fileName:%s open error\n", fileName);
			else
			{
			  fwrite(raw_data, sizeof(float), len, outfile);
			  fclose(outfile);
			}
		}
	}
	inline void df_write(Mat &blob,int layer_index, char const *path) const
	{
		int len = blob.total();
		const void* raw_data = blob.data;
		FILE * outfile;
		char fileName[100];
		sprintf(fileName, "%s/%d/%d-%d.bin", path, run_id, layer_index, 0);
		outfile = fopen(fileName, "wb" );
		if (outfile == NULL)
		  printf("fileName:%s open error\n", fileName);
		else
		{
		  fwrite(raw_data, sizeof(float), len, outfile);
		  fclose(outfile);
		}
	}
	
    char input_path[100];
    char output_path[100];
    int run_id = 0;
};

} // namespace ncnn

#endif // CAFFE_OUTPUT_BLOB_BIN_FILE
#endif // NCNN_DEBUG_MODEL_H
