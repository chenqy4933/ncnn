
#ifndef NCNN_MODEL_CAFFE_H
#define NCNN_MODEL_CAFFE_H

#include <stdio.h>
#include "mat.h"
#include "platform.h"
#include "caffe.pb.h"

struct _MTString {
	/* never reference these directly! */
	unsigned long size;	/* buffer size;  limit */
	unsigned long len;	/* current length  */
	unsigned char *buf;
};

typedef struct _MTString MTString;

union _u_parm{
	int parm_int;
	float parm_float;
};

typedef union _u_parm u_parm;

void
stringAppend(MTString * s, const void *str, int len);

namespace ncnn {

class Model_Caffe
{
public:

	static int caffe2ncnn(unsigned char** ppm,
		unsigned char** bpm,
        const char* caffemodel,
        const char* caffeproto = NULL);
	
	static int caffe2ncnn(unsigned char** ppm,
		unsigned char** bpm,
        const char* mergemodel_mem,
        int net_size);

	static int CaffeNetParameter2ncnn(unsigned char** ppm,
		unsigned char** bpm,
        caffe::NetParameter& proto,
        caffe::NetParameter& net,
        long proto_sz,
        long net_sz);

    Model_Caffe();
	
	static inline size_t alignSize(size_t sz, int n)
	{
	    return (sz + n-1) & -n;
	}

//    static inline void MTappend(MTString *mts, u_parm up)
//    {
//        stringAppend(mts,&up,sizeof(u_parm));
//    }
    
    static inline void MTappend(MTString *mts, int up)
    {
        stringAppend(mts,&up,sizeof(int));
    }
    
    static inline void MTappend(MTString *mts, float up)
    {
        stringAppend(mts,&up,sizeof(float));
    }

	static inline void MTappend(MTString *mts,char* str)
	{
		int len = strlen(str);
        stringAppend(mts,str,len);
		stringAppend(mts,"\0",1);
	}

protected:
	static unsigned short float2half(float value);
	static int quantize_weight(float *data, size_t data_length, std::vector<unsigned short>& float16_weights);
	static bool quantize_weight(float *data, size_t data_length, int quantize_level, std::vector<float> &quantize_table, std::vector<unsigned char> &quantize_index);
#ifndef NCNN_USE_PROTOBUF_LITE
	static bool read_proto_from_text(const char* filepath, google::protobuf::Message* message,long *size);
	static bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message,long *size);
#else
	static bool read_proto_from_text(const char* filepath, google::protobuf::MessageLite* message,long *size);
	static bool read_proto_from_binary(const char* filepath, google::protobuf::MessageLite* message,long *size);
#endif
	const unsigned char*& mem;
};

} // namespace ncnn

#endif // NCNN_MODELBIN_H
