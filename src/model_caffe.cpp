
#include <stdio.h>
#include <limits.h>
#include <math.h>

#include <fstream>
#include <set>
#include <limits>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifndef NCNN_USE_PROTOBUF_LITE
#include "google/protobuf/message.h"
#else
#include "google/protobuf/message_lite.h"
#endif

#include "caffe.pb.h"

#include "model_caffe.h"


//#include <string.h>
//#include <vector>
//#include "platform.h"
const MTString StringNull = { 0, 0, NULL };
const int Mem64k = 64*1024;


static void
stringInitBuf(MTString * s, size_t sz)
{
	s->buf = (unsigned char *)calloc(1, sz);
	s->size = sz;
    s->len = 0;
}

void
stringLimitInit(MTString * s, const void *str, int len)
{
	stringInitBuf(s, len);
	s->len = len;
	memcpy(s->buf, str, len);
}

void
stringInit(MTString * s, const void *str,int len)
{
	if (str)
		stringLimitInit(s, str, len);
	else
		*s = StringNull;
}

MTString
stringDup(const MTString * s,int len)
{
	MTString dup;
	stringInit(&dup, s->buf,len);
	return dup;
}

void
stringClean(MTString * s)
{
	if (s->buf)
		free(s->buf);
	*s = StringNull;
}

void
stringReset(MTString * s, const void *str,int len)
{
	stringClean(s);
	stringInit(s, str,len);
}

void
stringAppend(MTString * s, const void *str, int len)
{
	if (s->len + len < s->size) {
		memcpy(s->buf + s->len, str, len);
		s->len += len;
	} else {
		MTString snew = StringNull;
        long newlen = s->len + len;
        long extsize = (len > Mem64k?(len & 0xFFFF0000):Mem64k);
		stringInitBuf(&snew, newlen + extsize);
		if (s->len)
			memcpy(snew.buf, s->buf, s->len);
		if (len)
			memcpy(snew.buf + s->len, str, len);
        snew.len = newlen;
		stringClean(s);
		*s = snew;
	}
}


namespace ncnn {


// convert float to half precision floating point
unsigned short Model_Caffe::float2half(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (- 127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // underflow
            if (newexp >= -10)
            {
                // denormal half-precision
                unsigned short sig = (significand | 0x800000) >> (14 - newexp);
                fp16 = (sign << 15) | (0x00 << 10) | sig;
            }
            else
            {
                // underflow
                fp16 = (sign << 15) | (0x00 << 10) | 0x00;
            }
        }
        else
        {
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

int Model_Caffe::quantize_weight(float *data, size_t data_length, std::vector<unsigned short>& float16_weights)
{
    float16_weights.resize(data_length);

    for (size_t i = 0; i < data_length; i++)
    {
        float f = data[i];

        unsigned short fp16 = float2half(f);

        float16_weights[i] = fp16;
    }

    // magic tag for half-precision floating point
    return 0x01306B47;
}

bool Model_Caffe::quantize_weight(float *data, size_t data_length, int quantize_level, std::vector<float> &quantize_table, std::vector<unsigned char> &quantize_index) {

    assert(quantize_level != 0);
    assert(data != NULL);
    assert(data_length > 0);

    if (data_length < static_cast<size_t>(quantize_level)) {
        fprintf(stderr, "No need quantize,because: data_length < quantize_level");
        return false;
    }

    quantize_table.reserve(quantize_level);
    quantize_index.reserve(data_length);

    // 1. Find min and max value
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();

    for (size_t i = 0; i < data_length; ++i)
    {
        if (max_value < data[i]) max_value = data[i];
        if (min_value > data[i]) min_value = data[i];
    }
    float strides = (max_value - min_value) / quantize_level;

    // 2. Generate quantize table
    for (int i = 0; i < quantize_level; ++i)
    {
        quantize_table.push_back(min_value + i * strides);
    }

    // 3. Align data to the quantized value
    for (size_t i = 0; i < data_length; ++i)
    {
        size_t table_index = int((data[i] - min_value) / strides);
        table_index = std::min<float>(table_index, quantize_level - 1);

        float low_value  = quantize_table[table_index];
        float high_value = low_value + strides;

        // find a nearest value between low and high value.
        float targetValue = data[i] - low_value < high_value - data[i] ? low_value : high_value;

        table_index = int((targetValue - min_value) / strides);
        table_index = std::min<float>(table_index, quantize_level - 1);
        quantize_index.push_back(table_index);
    }

    return true;
}


bool Model_Caffe::read_proto_from_text(const char* filepath,
#ifndef NCNN_USE_PROTOBUF_LITE
    google::protobuf::Message* message,
#else
    google::protobuf::MessageLite* message,
#endif
    long *size)
{
#ifndef NCNN_USE_PROTOBUF_LITE
    std::ifstream fs(filepath, std::ifstream::in);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }
    fs.seekg(0,std::ifstream::end);
    *size = fs.tellg();
    fs.seekg(0,std::ifstream::beg);
    
    google::protobuf::io::IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, message);

    fs.close();

    return success;
#else
    return false;
#endif
}

bool Model_Caffe::read_proto_from_binary(const char* filepath,
#ifndef NCNN_USE_PROTOBUF_LITE
    google::protobuf::Message* message,
#else
    google::protobuf::MessageLite* message,
#endif
    long *size)
{
#ifndef NCNN_USE_PROTOBUF_LITE
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }
    fs.seekg(0,std::ifstream::end);
    *size = fs.tellg();
    fs.seekg(0,std::ifstream::beg);
    

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
#else
    return false;
#endif
}

int Model_Caffe::caffe2ncnn(unsigned char** ppm,
		unsigned char** bpm,
        const char* caffemodel,
        const char* caffeproto)
{
    // load
    if (caffeproto != NULL)
    {
        caffe::NetParameter proto;
        caffe::NetParameter net;
        long proto_size;
        long net_size;
    
        bool s0 = read_proto_from_text(caffeproto, &proto, &proto_size);
        if (!s0)
        {
            fprintf(stderr, "read_proto_from_text failed\n");
            return -1;
        }
        bool s1 = read_proto_from_binary(caffemodel, &net, &net_size);
        if (!s1)
        {
            fprintf(stderr, "read_proto_from_binary failed\n");
            return -1;
        }
        return CaffeNetParameter2ncnn(ppm,	bpm, proto, net,proto_size,net_size);
    }
    else
    {
        caffe::NetParameter net;
        long net_size;
        
        bool s1 = read_proto_from_binary(caffemodel, &net, &net_size);
        if (!s1)
        {
            fprintf(stderr, "read_proto_from_binary failed\n");
            return -1;
        }
        return CaffeNetParameter2ncnn(ppm,	bpm, net, net, net_size, Mem64k);
    }
}

int Model_Caffe::caffe2ncnn(unsigned char** ppm,
		unsigned char** bpm,
        const char* mergemodel_mem,
        int net_size)
{
    caffe::NetParameter net;
    net.ParseFromArray(mergemodel_mem, net_size);
    return CaffeNetParameter2ncnn(ppm,	bpm, net, net, net_size, Mem64k);
}

inline void binary16_decode(const uint16_t* f16, float* f32)
{
	uint32_t& u32 = *(uint32_t*)(f32);
	const uint16_t& u16 = *f16;
	int ex = ((u16 & 0x7c00) >> 10);
	ex = std::max(0, std::min(ex - 0x0f + 0x7f, 0xff));
	u32 = ((u16 & 0x8000) << 16) + (ex << 23) + ((u16 & 0x03ff) << 13);
}

inline Mat BlobProto2Mat(const caffe::BlobProto& blobProto)
{
    const char *data;
    int size;
    Mat mat;
    if(blobProto.has_byte_data()){
        const std::string byte_data = blobProto.byte_data();
        data = (const char *)byte_data.c_str();
        size = byte_data.size();
        mat.create(size/2);
        for (int i = 0; i < size/2; ++i) {
    	  float val;
    	  binary16_decode((const uint16_t*)(&data[2 * i]), &val);
    	  mat[i] = val;
    	}
    }else{
        data = (const char *)blobProto.data().data();
        size = blobProto.data_size();
        mat = Mat(size, (void *)data);
    }
    return mat;
}

int Model_Caffe::CaffeNetParameter2ncnn(unsigned char** ppm,
		unsigned char** bpm,
        caffe::NetParameter& proto,
        caffe::NetParameter& net,
        long proto_sz,
        long net_sz)
{
    int nolayerparam_layer = 0;
    int nolayerparam_blob = 0;
    //current only use quantize_level = 0
    int quantize_level = 0;
    char tmp[4096];

    if (quantize_level != 0 && quantize_level != 256 && quantize_level != 65536) {
        fprintf(stderr, "NCNN: only support quantize level = 0, 256, or 65536");
        return -1;
    }

    //std::vector<Mat> bp;
    MTString pp;
    stringInitBuf(&pp,proto_sz);
    MTString bp;
    stringInitBuf(&bp,net_sz);
    int parm_int;
    
    // magic
    //std::string pp;
    //sprintf(pp, "7767517\n");
    //pp += "7767517\n";
    parm_int = 7767517;
    stringAppend(&pp,&parm_int,sizeof(int));
    
    // rename mapping for identical bottom top style
    std::map<std::string, std::string> blob_name_decorated;

    // bottom blob reference
    std::map<std::string, int> bottom_reference;

    // global definition line
    // [layer count] [blob count]
    int layer_count = proto.layer_size();
    std::set<std::string> blob_names;
    for (int i=0; i<layer_count; i++)
    {
        const caffe::LayerParameter& layer = proto.layer(i);

        for (int j=0; j<layer.bottom_size(); j++)
        {
            std::string blob_name = layer.bottom(j);
            if (blob_name_decorated.find(blob_name) != blob_name_decorated.end())
            {
                blob_name = blob_name_decorated[blob_name];
            }

            blob_names.insert(blob_name);

            if (bottom_reference.find(blob_name) == bottom_reference.end())
            {
                bottom_reference[blob_name] = 1;
            }
            else
            {
                bottom_reference[blob_name] = bottom_reference[blob_name] + 1;
            }
        }

        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = layer.top(0) + "_" + layer.name();
            blob_name_decorated[layer.top(0)] = blob_name;
            blob_names.insert(blob_name);
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
                blob_names.insert(blob_name);
            }
        }
    }

    if(proto.input_size())
    {
        std::string blob_name = proto.input(0);
        nolayerparam_layer += 1;
        if (bottom_reference.find(blob_name) == bottom_reference.end())
        {
            nolayerparam_blob += 1;
            fprintf(stderr, "input %s is no used.\n", blob_name.c_str());
        }
    }
    
    // remove bottom_reference entry with reference equals to one
    int splitncnn_blob_count = 0;
    std::map<std::string, int>::iterator it = bottom_reference.begin();
    while (it != bottom_reference.end())
    {
        if (it->second == 1)
        {
            bottom_reference.erase(it++);
        }
        else
        {
            splitncnn_blob_count += it->second;
//             fprintf(stderr, "%s %d\n", it->first.c_str(), it->second);
            ++it;
        }
    }
    
    parm_int = layer_count + bottom_reference.size() + nolayerparam_layer;
    MTappend(&pp,parm_int);
    parm_int = blob_names.size() + splitncnn_blob_count + nolayerparam_blob;
    MTappend(&pp,parm_int);

    if(proto.input_size())
    {
        std::string blob_name = proto.input(0);
        sprintf(tmp, "%s", "Input");
        MTappend(&pp,tmp); 
        sprintf(tmp, "nolayerparam_%s", blob_name.c_str());
        MTappend(&pp,tmp); 
        MTappend(&pp,0);
        MTappend(&pp,1);
        sprintf(tmp, "%s", blob_name.c_str());
        MTappend(&pp,tmp);
        if (proto.input_dim_size() == 4)
        {
            MTappend(&pp,0);
            MTappend(&pp,(int)proto.input_dim(3));
            MTappend(&pp,1);
            MTappend(&pp,(int)proto.input_dim(2));
            MTappend(&pp,2);
            MTappend(&pp,(int)proto.input_dim(1));
        }
        else if (proto.input_dim_size() == 3)
        {
            MTappend(&pp,0);
            MTappend(&pp,(int)proto.input_dim(2));
            MTappend(&pp,1);
            MTappend(&pp,(int)proto.input_dim(1));
            MTappend(&pp,2);
            MTappend(&pp,-233);
        }
        else if (proto.input_dim_size() == 2)
        {
            MTappend(&pp,0);
            MTappend(&pp,(int)proto.input_dim(1));
            MTappend(&pp,1);
            MTappend(&pp,-233);
            MTappend(&pp,2);
            MTappend(&pp,-233);
        }
        MTappend(&pp,-233);
    }

    // populate
    blob_name_decorated.clear();
    int internal_split = 0;
    for (int i=0; i<layer_count; i++)
    {
        const caffe::LayerParameter& layer = proto.layer(i);

        // layer definition line, repeated
        // [type] [name] [bottom blob count] [top blob count] [bottom blobs] [top blobs] [layer specific params]
        if (layer.type() == "Convolution")
        {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            if (convolution_param.group() != 1){
                MTappend(&pp,(char *)"ConvolutionDepthWise");
            }else{
                //stringAppend(&pp,"Convolution\0",strlen("Convolution\0"));
                MTappend(&pp,(char *)"Convolution");
            }
        }
        else if (layer.type() == "ConvolutionDepthwise")
        {
            MTappend(&pp,(char *)"ConvolutionDepthWise");
        }
        else if (layer.type() == "Deconvolution")
        {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            if (convolution_param.group() != 1){
                MTappend(&pp,(char *)"DeconvolutionDepthWise");
            }else{
                MTappend(&pp,(char *)"Deconvolution");
            }
        }
        else if (layer.type() == "MemoryData")
        {
            MTappend(&pp,(char *)"Input");
        }
        else if (layer.type() == "Python")
        {
            const caffe::PythonParameter& python_param = layer.python_param();
            std::string python_layer_name = python_param.layer();
            if (python_layer_name == "ProposalLayer"){
                MTappend(&pp,(char *)"Proposal");
            }else{
                sprintf(tmp, "%s", python_layer_name.c_str());
                MTappend(&pp,tmp);
            }
        }
        else
        {
            sprintf(tmp, "%s", layer.type().c_str());
            MTappend(&pp,tmp);
        }
        sprintf(tmp, "%s", layer.name().c_str());
        MTappend(&pp,tmp);
        MTappend(&pp,layer.bottom_size());
        MTappend(&pp,layer.top_size());
        
        for (int j=0; j<layer.bottom_size(); j++)
        {
            std::string blob_name = layer.bottom(j);
            if (blob_name_decorated.find(layer.bottom(j)) != blob_name_decorated.end())
            {
                blob_name = blob_name_decorated[layer.bottom(j)];
            }

            if (bottom_reference.find(blob_name) != bottom_reference.end())
            {
                int refidx = bottom_reference[blob_name] - 1;
                bottom_reference[blob_name] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                blob_name = blob_name + splitsuffix;
            }
            sprintf(tmp, "%s", blob_name.c_str());
            MTappend(&pp,tmp);
        }

        // if input = output, rename: blobname --> blobname_layername
        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = layer.top(0) + "_" + layer.name();
            blob_name_decorated[layer.top(0)] = blob_name;

            sprintf(tmp, "%s", blob_name.c_str());
            MTappend(&pp,tmp);
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
                sprintf(tmp, "%s", blob_name.c_str());
                MTappend(&pp,tmp);
            }
        }

        // find blob binary by layer name
        int netidx;
        for (netidx=0; netidx<net.layer_size(); netidx++)
        {
            if (net.layer(netidx).name() == layer.name())
            {
                break;
            }
        }

        // layer specific params
        if (layer.type() == "BatchNorm")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& mean_blob = binlayer.blobs(0);
            Mat mean_mat = BlobProto2Mat(mean_blob);
            const caffe::BlobProto& var_blob = binlayer.blobs(1);
            Mat var_mat = BlobProto2Mat(var_blob);

            MTappend(&pp,0);
            MTappend(&pp,(int)mean_mat.total());

            const caffe::BatchNormParameter& batch_norm_param = layer.batch_norm_param();
            float eps = batch_norm_param.eps();

            std::vector<float> ones(mean_mat.total(), 1.f);
            stringAppend(&bp, ones.data(), sizeof(float) * ones.size());// slope

            if (binlayer.blobs_size() < 3)
            {
                stringAppend(&bp, mean_mat.data, sizeof(float) * mean_mat.total());
                float tmp;
                for (int j=0; j<(int)var_mat.total(); j++)
                {
                    tmp = var_mat[j] + eps;
                    stringAppend(&bp, &tmp, sizeof(float) * 1);
                }
            }
            else
            {
                float scale_factor = 1 / binlayer.blobs(2).data().data()[0];
                // premultiply scale_factor to mean and variance
                float tmp;
                for (int j=0; j<(int)mean_mat.total(); j++)
                {
                    tmp = mean_mat[j] * scale_factor;
                    stringAppend(&bp, &tmp, sizeof(float) * 1);
                }
                for (int j=0; j<(int)var_mat.total(); j++)
                {
                    tmp = var_mat[j] * scale_factor + eps;
                    stringAppend(&bp, &tmp, sizeof(float) * 1);
                }
            }

            std::vector<float> zeros(mean_mat.total(), 0.f);
            stringAppend(&bp, zeros.data(), sizeof(float) * zeros.size());// bias
        }
        else if (layer.type() == "Concat")
        {
            const caffe::ConcatParameter& concat_param = layer.concat_param();
            int dim = concat_param.axis() - 1;
            MTappend(&pp,0);
            MTappend(&pp,dim);
        }
        else if (layer.type() == "Convolution" || layer.type() == "ConvolutionDepthwise")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            Mat weight_mat = BlobProto2Mat(weight_blob);
            
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            MTappend(&pp,0);
            MTappend(&pp,(int)convolution_param.num_output());
            
            if (convolution_param.has_kernel_w() && convolution_param.has_kernel_h())
            {
                MTappend(&pp,1);
                MTappend(&pp,(int)convolution_param.kernel_w());
                MTappend(&pp,11);
                MTappend(&pp,(int)convolution_param.kernel_h());
            }
            else
            {
                MTappend(&pp,1);
                MTappend(&pp,(int)convolution_param.kernel_size(0));
            }
            MTappend(&pp,2);
            parm_int = convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1;
            MTappend(&pp,parm_int);
            if (convolution_param.has_stride_w() && convolution_param.has_stride_h())
            {
                MTappend(&pp,3);
                MTappend(&pp,(int)convolution_param.stride_w());
                MTappend(&pp,13);
                MTappend(&pp,(int)convolution_param.stride_h());
            }
            else
            {
                MTappend(&pp,3);
                parm_int = convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1;
                MTappend(&pp,parm_int);
            }
            if (convolution_param.has_pad_w() && convolution_param.has_pad_h())
            {
                MTappend(&pp,4);
                parm_int = convolution_param.pad_w();
                MTappend(&pp,parm_int);
                MTappend(&pp,14);
                parm_int = convolution_param.pad_h();
                MTappend(&pp,parm_int);
            }
            else
            {
                MTappend(&pp,4);
                parm_int = convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0;
                MTappend(&pp,parm_int);
            }
            MTappend(&pp,5);
            MTappend(&pp,(int)convolution_param.bias_term());
            MTappend(&pp,6);
            MTappend(&pp,(int)weight_mat.total());

            if (layer.type() == "ConvolutionDepthwise")
            {
                MTappend(&pp,7);
                MTappend(&pp,(int)convolution_param.num_output());
            }
            else if (convolution_param.group() != 1)
            {
                MTappend(&pp,7);
                MTappend(&pp,(int)convolution_param.group());
            }

            for (int j = 0; j < binlayer.blobs_size(); j++)
            {
                int quantize_tag = 0;
                const caffe::BlobProto& blob = binlayer.blobs(j);
                Mat blob_mat = BlobProto2Mat(blob);
                
                std::vector<float> quantize_table;
                std::vector<unsigned char> quantize_index;

                std::vector<unsigned short> float16_weights;

                // we will not quantize the bias values
                if (j == 0 && quantize_level != 0)
                {
                    if (quantize_level == 256)
                    {
                    quantize_tag = quantize_weight((float *)blob_mat.data, blob_mat.total(), quantize_level, quantize_table, quantize_index);
                    }
                    else if (quantize_level == 65536)
                    {
                    quantize_tag = quantize_weight((float *)blob_mat.data, blob_mat.total(), float16_weights);
                    }
                }

                // write quantize tag first
                if (j == 0){
                    /*
                    Mat m(1);
                    if (m.empty())
                        return m;
                    nread = memcpy(m, &quantize_tag,1 * sizeof(float));
                    bp.push_back(m);
                    */
                    stringAppend(&bp,&quantize_tag, 1 * sizeof(float));
                }

                if (quantize_tag)
                { 
                    int p0 = 0;
                    if (quantize_level == 256)
                    {
                    // write quantize table and index
                    stringAppend(&bp, quantize_table.data(), quantize_table.size() * sizeof(float));
                    stringAppend(&bp, quantize_index.data(), quantize_index.size() * sizeof(unsigned char));
                    p0 += quantize_table.size() * sizeof(float);
                    p0 += quantize_index.size() * sizeof(unsigned char);
                    }
                    else if (quantize_level == 65536)
                    {
                    stringAppend(&bp, float16_weights.data(), sizeof(unsigned short) * float16_weights.size());
                    p0 += sizeof(unsigned short) * float16_weights.size();
                    }
                    // padding to 32bit align
                    int nwrite = p0;
                    int nalign = alignSize(nwrite, 4);
                    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
                    stringAppend(&bp, padding, sizeof(unsigned char) * (nalign - nwrite)); 
                }
                else
                {
                    // write original data
                    /*
                    int w = blob.data_size();
                    Mat m(w);
                    if (m.empty())
                        return m;
                    nread = memcpy(m, blob.data().data(), w * sizeof(float));
                    bp.push_back(m);
                    */
                    stringAppend(&bp, blob_mat.data, sizeof(float) * blob_mat.total());
                    
                }
            }

        }
        else if (layer.type() == "Crop")
        {
            const caffe::CropParameter& crop_param = layer.crop_param();
            int num_offset = crop_param.offset_size();
            int woffset = (num_offset == 2) ? crop_param.offset(0) : 0;
            int hoffset = (num_offset == 2) ? crop_param.offset(1) : 0;
            MTappend(&pp,0);
            MTappend(&pp,woffset);
            MTappend(&pp,1);
            MTappend(&pp,hoffset);
        }
        else if (layer.type() == "Deconvolution")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            Mat weight_mat = BlobProto2Mat(weight_blob);
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            MTappend(&pp,0);
            MTappend(&pp,(int)convolution_param.num_output());
            if (convolution_param.has_kernel_w() && convolution_param.has_kernel_h())
            {
                MTappend(&pp,1);
                MTappend(&pp,(int)convolution_param.kernel_w());
                MTappend(&pp,11);
                MTappend(&pp,(int)convolution_param.kernel_h());
            }
            else
            {
                MTappend(&pp,1);
                MTappend(&pp,(int)convolution_param.kernel_size(0));
            }
            parm_int = 2;
            MTappend(&pp,2);
            parm_int = convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1;
            MTappend(&pp,parm_int);
            if (convolution_param.has_stride_w() && convolution_param.has_stride_h())
            {
                MTappend(&pp,3);
                MTappend(&pp,(int)convolution_param.stride_w());
                MTappend(&pp,13);
                MTappend(&pp,(int)convolution_param.stride_h());
            }
            else
            {
                MTappend(&pp, 3);
                parm_int = convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1;
                MTappend(&pp, parm_int);
            }
            if (convolution_param.has_pad_w() && convolution_param.has_pad_h())
            {
                MTappend(&pp,4);
                MTappend(&pp,(int)convolution_param.pad_w());
                MTappend(&pp,14);
                MTappend(&pp,(int)convolution_param.pad_h());
            }
            else
            {
                MTappend(&pp, 4);
                parm_int = convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0;
                MTappend(&pp, parm_int);
            }
            MTappend(&pp,5);
            MTappend(&pp,(int)convolution_param.bias_term());
            MTappend(&pp,6);
            MTappend(&pp,(int)weight_mat.total());

            if (convolution_param.group() != 1)
            {
                MTappend(&pp,7);
                MTappend(&pp,(int)convolution_param.group());
            }

            int quantized_weight = 0;
            stringAppend(&bp, &quantized_weight, sizeof(int)* 1);
            
            // reorder weight from inch-outch to outch-inch
            int ksize ;
            if (convolution_param.has_kernel_w() && convolution_param.has_kernel_h())
            {
                ksize = convolution_param.kernel_w() * convolution_param.kernel_h();
            }
            else
            {
                ksize = convolution_param.kernel_size(0) * convolution_param.kernel_size(0);
            }
            int num_output = convolution_param.num_output();
            int num_input = weight_mat.total() / (ksize) / num_output;
            float* weight_data_ptr = (float *)weight_mat.data;
            for (int k=0; k<num_output; k++)
            {
                for (int j=0; j<num_input; j++)
                {
                    stringAppend(&bp, weight_data_ptr + (j*num_output + k) * ksize, sizeof(float) * ksize);
                }
            }

            for (int j=1; j<binlayer.blobs_size(); j++)
            {
                const caffe::BlobProto& blob = binlayer.blobs(j);
                Mat blob_mat = BlobProto2Mat(blob);
                stringAppend(&bp, blob_mat.data, sizeof(float) * blob_mat.total());
            }
        }
        #if 0
        else if (layer.type() == "DetectionOutput")
        {
            const caffe::DetectionOutputParameter& detection_output_param = layer.detection_output_param();
            const caffe::NonMaximumSuppressionParameter& nms_param = detection_output_param.nms_param();
            sprintf(tmp, " 0=%d 1=%f 2=%d 3=%d 4=%f", detection_output_param.num_classes()
            , nms_param.nms_threshold()
            , nms_param.top_k()
            , detection_output_param.keep_top_k()
            , detection_output_param.confidence_threshold());
            pp += tmp;
        }
        else if (layer.type() == "Dropout")
        {
            const caffe::DropoutParameter& dropout_param = layer.dropout_param();
            if (dropout_param.has_scale_train() && !dropout_param.scale_train())
            {
                float scale = 1.f - dropout_param.dropout_ratio();
                sprintf(tmp, " 0=%f", scale);
                pp += tmp;
            }
        }
        #endif
        else if (layer.type() == "Eltwise")
        {
            const caffe::EltwiseParameter& eltwise_param = layer.eltwise_param();
            int coeff_size = eltwise_param.coeff_size();
            MTappend(&pp,0);
            MTappend(&pp,(int)eltwise_param.operation());
            MTappend(&pp,-23301);
            MTappend(&pp,(int)coeff_size);
            for (int j=0; j<coeff_size; j++)
            {
                MTappend(&pp,(float)eltwise_param.coeff(j));
            }
        }
        else if (layer.type() == "ELU")
        {
            const caffe::ELUParameter& elu_param = layer.elu_param();
            MTappend(&pp,0);
            MTappend(&pp,(float)elu_param.alpha());
        }
        else if (layer.type() == "InnerProduct")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            Mat weight_mat = BlobProto2Mat(weight_blob);
            const caffe::InnerProductParameter& inner_product_param = layer.inner_product_param();
            MTappend(&pp,0);
            MTappend(&pp,(int)inner_product_param.num_output());
            MTappend(&pp,1);
            MTappend(&pp,(int)inner_product_param.bias_term());
            MTappend(&pp,2);
            MTappend(&pp,(int)weight_mat.total());

            for (int j=0; j<binlayer.blobs_size(); j++)
            {
                int quantize_tag = 0;
                const caffe::BlobProto& blob = binlayer.blobs(j);
                Mat blob_mat = BlobProto2Mat(blob);
                std::vector<float> quantize_table;
                std::vector<unsigned char> quantize_index;

                std::vector<unsigned short> float16_weights;

                // we will not quantize the bias values
                if (j == 0 && quantize_level != 0)
                {
                    if (quantize_level == 256)
                    {
                    quantize_tag = quantize_weight((float *)blob_mat.data, blob_mat.total(), quantize_level, quantize_table, quantize_index);
                    }
                    else if (quantize_level == 65536)
                    {
                    quantize_tag = quantize_weight((float *)blob_mat.data, blob_mat.total(), float16_weights);
                    }
                }

                // write quantize tag first
                if (j == 0)
                    stringAppend(&bp, &quantize_tag, sizeof(int) * 1);

                if (quantize_tag)
				{
                    int p0 = 0;
                    if (quantize_level == 256)
                    {
                    // write quantize table and index
                    stringAppend(&bp, quantize_table.data(), sizeof(float) * quantize_table.size());
                    stringAppend(&bp, quantize_index.data(), sizeof(unsigned char) * quantize_index.size());
                    p0 += sizeof(float) * quantize_table.size();
                    p0 += sizeof(unsigned char) * quantize_index.size();
                    }
                    else if (quantize_level == 65536)
                    {
                    stringAppend(&bp, float16_weights.data(), sizeof(unsigned short) * float16_weights.size());
                    p0 += sizeof(unsigned short) * float16_weights.size();
                    }
                    // padding to 32bit align
                    int nwrite = p0;
                    int nalign = alignSize(nwrite, 4);
                    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
                    stringAppend(&bp, padding, sizeof(unsigned char) * (nalign - nwrite));
                }
                else
				{
                    // write original data
                    stringAppend(&bp, blob_mat.data, sizeof(float) * blob_mat.total());
                }
            }
        }
        else if (layer.type() == "Input")
        {
            const caffe::InputParameter& input_param = layer.input_param();
            const caffe::BlobShape& bs = input_param.shape(0);
            if (bs.dim_size() == 4)
            {
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(3));
                MTappend(&pp,1);
                MTappend(&pp,(int)bs.dim(2));
                MTappend(&pp,2);
                MTappend(&pp,(int)bs.dim(1));
            }
            else if (bs.dim_size() == 3)
            {
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(2));
                MTappend(&pp,1);
                MTappend(&pp,(int)bs.dim(1));
                MTappend(&pp,2);
                MTappend(&pp,-233);
            }
            else if (bs.dim_size() == 2)
            {
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(1));
                MTappend(&pp,1);
                MTappend(&pp,-233);
                MTappend(&pp,2);
                MTappend(&pp,-233);
            }
        }
        else if (layer.type() == "Interp")
        {
            const caffe::InterpParameter& interp_param = layer.interp_param();
            MTappend(&pp,0);
            MTappend(&pp,2);
            MTappend(&pp,1);
            MTappend(&pp,(float)interp_param.zoom_factor());
            MTappend(&pp,2);
            MTappend(&pp,(float)interp_param.zoom_factor());
            MTappend(&pp,3);
            MTappend(&pp,(int)interp_param.height());
            MTappend(&pp,4);
            MTappend(&pp,(int)interp_param.width());
        }
        else if (layer.type() == "LRN")
        {
            const caffe::LRNParameter& lrn_param = layer.lrn_param();
            MTappend(&pp,0);
            parm_int = lrn_param.norm_region();
            MTappend(&pp,parm_int);
            MTappend(&pp,1);
            MTappend(&pp,(int)lrn_param.local_size());
            MTappend(&pp,2);
            MTappend(&pp,(float)lrn_param.alpha());
            MTappend(&pp,3);
            MTappend(&pp,(float)lrn_param.beta());
        }
        else if (layer.type() == "MemoryData")
        {
            const caffe::MemoryDataParameter& memory_data_param = layer.memory_data_param();
            MTappend(&pp,0);
            MTappend(&pp,(int)memory_data_param.width());
            MTappend(&pp,1);
            MTappend(&pp,(int)memory_data_param.height());
            MTappend(&pp,2);
            MTappend(&pp,(int)memory_data_param.channels());
        }
        else if (layer.type() == "MVN")
        {
            const caffe::MVNParameter& mvn_param = layer.mvn_param();
            MTappend(&pp,0);
            MTappend(&pp,mvn_param.normalize_variance());
            MTappend(&pp,1);
            MTappend(&pp,mvn_param.across_channels());
            MTappend(&pp,2);
            MTappend(&pp,(float)mvn_param.eps());
        }
        #if 0
        else if (layer.type() == "Normalize")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);
            const caffe::BlobProto& scale_blob = binlayer.blobs(0);
            const caffe::NormalizeParameter& norm_param = layer.norm_param();
            sprintf(tmp, " 0=%d 1=%d 2=%f 3=%d", norm_param.across_spatial()
                , norm_param.channel_shared()
                , norm_param.eps()
                , scale_blob.data_size());
            pp += tmp;

            stringAppend(&bp, scale_blob.data().data(), sizeof(float) * scale_blob.data_size());
        }
        else if (layer.type() == "Permute")
        {
            const caffe::PermuteParameter& permute_param = layer.permute_param();
            int order_size = permute_param.order_size();
            int order_type = 0;
            if (order_size == 0)
                order_type = 0;
            if (order_size == 1)
            {
                int order0 = permute_param.order(0);
                if (order0 == 0)
                    order_type = 0;
                // permute with N not supported
            }
            if (order_size == 2)
            {
                int order0 = permute_param.order(0);
                int order1 = permute_param.order(1);
                if (order0 == 0)
                {
                    if (order1 == 1) // 0 1 2 3
                        order_type = 0;
                    else if (order1 == 2) // 0 2 1 3
                        order_type = 2;
                    else if (order1 == 3) // 0 3 1 2
                        order_type = 4;
                }
                // permute with N not supported
            }
            if (order_size == 3 || order_size == 4)
            {
                int order0 = permute_param.order(0);
                int order1 = permute_param.order(1);
                int order2 = permute_param.order(2);
                if (order0 == 0)
                {
                    if (order1 == 1)
                    {
                        if (order2 == 2) // 0 1 2 3
                            order_type = 0;
                        if (order2 == 3) // 0 1 3 2
                            order_type = 1;
                    }
                    else if (order1 == 2)
                    {
                        if (order2 == 1) // 0 2 1 3
                            order_type = 2;
                        if (order2 == 3) // 0 2 3 1
                            order_type = 3;
                    }
                    else if (order1 == 3)
                    {
                        if (order2 == 1) // 0 3 1 2
                            order_type = 4;
                        if (order2 == 2) // 0 3 2 1
                            order_type = 5;
                    }
                }
                // permute with N not supported
            }
            sprintf(tmp, " 0=%d", order_type);
            pp += tmp;
        }
        #endif
        else if (layer.type() == "Pooling")
        {
            const caffe::PoolingParameter& pooling_param = layer.pooling_param();
            MTappend(&pp,0);
            MTappend(&pp,pooling_param.pool()); 
            if (pooling_param.has_kernel_w() && pooling_param.has_kernel_h())
            {
                MTappend(&pp,1);
                MTappend(&pp,(int)pooling_param.kernel_w());
                MTappend(&pp,11);
                MTappend(&pp,(int)pooling_param.kernel_h());
            }
            else
            {
                MTappend(&pp,1);
                MTappend(&pp,(int)pooling_param.kernel_size());
            }
            if (pooling_param.has_stride_w() && pooling_param.has_stride_h())
            {
                MTappend(&pp,2);
                MTappend(&pp,(int)pooling_param.stride_w());
                MTappend(&pp,12);
                MTappend(&pp,(int)pooling_param.stride_h());
            }
            else
            {
                MTappend(&pp,2);
                MTappend(&pp,(int)pooling_param.stride());
            }
            if (pooling_param.has_pad_w() && pooling_param.has_pad_h())
            {
                MTappend(&pp,3);
                MTappend(&pp,(int)pooling_param.pad_w());
                MTappend(&pp,13);
                MTappend(&pp,(int)pooling_param.pad_h());
            }
            else
            {
                MTappend(&pp,3);
                MTappend(&pp,(int)pooling_param.pad());
            }
            MTappend(&pp,4);
            MTappend(&pp,pooling_param.has_global_pooling() ? pooling_param.global_pooling() : 0);
        }
        else if (layer.type() == "Upsample")
        {
            const caffe::UpsampleParameter &upsample_param = layer.upsample_param();
            if (upsample_param.has_scale())
            {
                MTappend(&pp, 0);
                MTappend(&pp, (int)upsample_param.scale());
            }
            else if (upsample_param.has_scale_h() && upsample_param.has_scale_w())
            {
                MTappend(&pp, 1);
                MTappend(&pp, (int)upsample_param.scale_w());
                MTappend(&pp, 11);
                MTappend(&pp, (int)upsample_param.scale_h());
            }
            else //if(upsample_param.has_upsample_w() && upsample_param.has_upsample_h())
            {
                if (upsample_param.has_upsample_w() && upsample_param.has_upsample_h())
                {
                    MTappend(&pp, 2);
                    MTappend(&pp, (int)upsample_param.upsample_w());
                    MTappend(&pp, 12);
                    MTappend(&pp, (int)upsample_param.upsample_h());
                }
            }
            if (upsample_param.has_pad_out_w() && upsample_param.has_pad_out_h())
            {
                MTappend(&pp, 3);
                MTappend(&pp, (int)upsample_param.pad_out_w());
                MTappend(&pp, 13);
                MTappend(&pp, (int)upsample_param.pad_out_h());
            }
        }
        
        else if (layer.type() == "Power")
        {
            const caffe::PowerParameter& power_param = layer.power_param();
            MTappend(&pp,0);
            MTappend(&pp,(float)power_param.power());
            MTappend(&pp,1);
            MTappend(&pp,(float)power_param.scale());
            MTappend(&pp,2);
            MTappend(&pp,(float)power_param.shift());
                
        }
        else if (layer.type() == "PReLU")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);
            const caffe::BlobProto& slope_blob = binlayer.blobs(0);
            Mat slope_mat = BlobProto2Mat(slope_blob);
            MTappend(&pp,0);
            MTappend(&pp, (int)slope_mat.total());
            stringAppend(&bp, slope_mat.data, sizeof(float) * slope_mat.total());
        }
        #if 0
        else if (layer.type() == "PriorBox")
        {
            const caffe::PriorBoxParameter& prior_box_param = layer.prior_box_param();

            int num_aspect_ratio = prior_box_param.aspect_ratio_size();
            for (int j=0; j<prior_box_param.aspect_ratio_size(); j++)
            {
                float ar = prior_box_param.aspect_ratio(j);
                if (fabs(ar - 1.) < 1e-6) {
                    num_aspect_ratio--;
                }
            }

            float variances[4] = {0.1f, 0.1f, 0.1f, 0.1f};
            if (prior_box_param.variance_size() == 4)
            {
                variances[0] = prior_box_param.variance(0);
                variances[1] = prior_box_param.variance(1);
                variances[2] = prior_box_param.variance(2);
                variances[3] = prior_box_param.variance(3);
            }
            else if (prior_box_param.variance_size() == 1)
            {
                variances[0] = prior_box_param.variance(0);
                variances[1] = prior_box_param.variance(0);
                variances[2] = prior_box_param.variance(0);
                variances[3] = prior_box_param.variance(0);
            }

            int flip = prior_box_param.has_flip() ? prior_box_param.flip() : 1;
            int clip = prior_box_param.has_clip() ? prior_box_param.clip() : 0;
            int image_width = -233;
            int image_height = -233;
            if (prior_box_param.has_img_size())
            {
                image_width = prior_box_param.img_size();
                image_height = prior_box_param.img_size();
            }
            else if (prior_box_param.has_img_w() && prior_box_param.has_img_h())
            {
                image_width = prior_box_param.img_w();
                image_height = prior_box_param.img_h();
            }

            float step_width = -233;
            float step_height = -233;
            if (prior_box_param.has_step())
            {
                step_width = prior_box_param.step();
                step_height = prior_box_param.step();
            }
            else if (prior_box_param.has_step_w() && prior_box_param.has_step_h())
            {
                step_width = prior_box_param.step_w();
                step_height = prior_box_param.step_h();
            }

            sprintf(tmp, " -23300=%d", prior_box_param.min_size_size());
            pp += tmp;
            for (int j=0; j<prior_box_param.min_size_size(); j++)
            {
                sprintf(tmp, ",%f", prior_box_param.min_size(j));
                pp += tmp;
            }
            sprintf(tmp, " -23301=%d", prior_box_param.max_size_size());
            pp += tmp;
            for (int j=0; j<prior_box_param.max_size_size(); j++)
            {
                sprintf(tmp, ",%f", prior_box_param.max_size(j));
                pp += tmp;
            }
            sprintf(tmp, " -23302=%d", num_aspect_ratio);
            pp += tmp;
            for (int j=0; j<prior_box_param.aspect_ratio_size(); j++)
            {
                float ar = prior_box_param.aspect_ratio(j);
                if (fabs(ar - 1.) < 1e-6) {
                    continue;
                }
                sprintf(tmp, ",%f", ar);
                pp += tmp;
            }
            sprintf(tmp, " 3=%f 4=%f 5=%f 6=%f 7=%d 8=%d 9=%d 10=%d 11=%f 12=%f 13=%f"
                , variances[0]
                , variances[1]
                , variances[2]
                , variances[3]
                , flip
                , clip
                , image_width
                , image_height
                , step_width
                , step_height
                , prior_box_param.offset());
            pp += tmp;
        }
        #endif
        else if (layer.type() == "Python")
        {
            const caffe::PythonParameter& python_param = layer.python_param();
            std::string python_layer_name = python_param.layer();
            if (python_layer_name == "ProposalLayer")
            {
                int feat_stride = 16;
                sscanf(python_param.param_str().c_str(), "'feat_stride': %d", &feat_stride);

                int base_size = 16;
//                 float ratio;
//                 float scale;
                int pre_nms_topN = 6000;
                int after_nms_topN = 300;
                float nms_thresh = 0.7;
                int min_size = 16;
                MTappend(&pp,0);
                MTappend(&pp,feat_stride);
                MTappend(&pp,1);
                MTappend(&pp,base_size);
                MTappend(&pp,2);
                MTappend(&pp,pre_nms_topN);
                MTappend(&pp,3);
                MTappend(&pp,after_nms_topN);
                MTappend(&pp,4);
                MTappend(&pp,nms_thresh);
                MTappend(&pp,5);
                MTappend(&pp, min_size);
            }
        }
        else if (layer.type() == "ReLU")
        {
            const caffe::ReLUParameter& relu_param = layer.relu_param();
            if (relu_param.has_negative_slope())
            {
                MTappend(&pp,0);
                MTappend(&pp,(float)relu_param.negative_slope());
            }
        }
        else if (layer.type() == "Reshape")
        {
            const caffe::ReshapeParameter& reshape_param = layer.reshape_param();
            const caffe::BlobShape& bs = reshape_param.shape();
            if (bs.dim_size() == 1)
            {
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(0));
                MTappend(&pp,1);
                MTappend(&pp,-233);
                MTappend(&pp,2);
                MTappend(&pp,-233);
            }
            else if (bs.dim_size() == 2)
            {
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(1));
                MTappend(&pp,1);
                MTappend(&pp,(int)bs.dim(0));
                MTappend(&pp,2);
                MTappend(&pp,-233);
            }
            else if (bs.dim_size() == 3)
            {
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(2));
                MTappend(&pp,1);
                MTappend(&pp,(int)bs.dim(1));
                MTappend(&pp,2);
                MTappend(&pp,(int)bs.dim(0));
            }
            else // bs.dim_size() == 4
            {
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(3));
                MTappend(&pp,1);
                MTappend(&pp,(int)bs.dim(2));
                MTappend(&pp,2);
                MTappend(&pp,(int)bs.dim(1));
            }
            MTappend(&pp,3);
            MTappend(&pp,0);
        }
        #if 0
        else if (layer.type() == "ROIPooling")
        {
            const caffe::ROIPoolingParameter& roi_pooling_param = layer.roi_pooling_param();
            sprintf(tmp, " 0=%d 1=%d 2=%f", roi_pooling_param.pooled_w()
                , roi_pooling_param.pooled_h());
                , roi_pooling_param.spatial_scale());
            pp += tmp;
        }
        #endif
        else if (layer.type() == "Scale")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::ScaleParameter& scale_param = layer.scale_param();
            bool scale_weight = scale_param.bias_term() ? (binlayer.blobs_size() == 2) : (binlayer.blobs_size() == 1);
            if (scale_weight)
            {
                const caffe::BlobProto& weight_blob = binlayer.blobs(0);
                Mat weight_mat = BlobProto2Mat(weight_blob);
                MTappend(&pp,0);
                MTappend(&pp,(int)weight_mat.total());
            }
            else
            {
                MTappend(&pp,0);
                MTappend(&pp,-233);
            }

            MTappend(&pp,1);
            MTappend(&pp,scale_param.bias_term());

            for (int j=0; j<binlayer.blobs_size(); j++)
            {
                const caffe::BlobProto& blob = binlayer.blobs(j);
                Mat blob_mat = BlobProto2Mat(blob);
                stringAppend(&bp, blob_mat.data, sizeof(float) * blob_mat.total());
            }
        }
        else if (layer.type() == "ShuffleChannel")
        {
            const caffe::ShuffleChannelParameter&
                    shuffle_channel_param = layer.shuffle_channel_param();
            MTappend(&pp,0);
            MTappend(&pp,(int)shuffle_channel_param.group());
        }
        else if (layer.type() == "Slice")
        {
            const caffe::SliceParameter& slice_param = layer.slice_param();
            //NCNN:  slice_dim means slice_point is no avalabe
            //MEITU: slice_dim means slice_point is avalabe
            if (!slice_param.has_slice_dim())
            {
                int num_slice = layer.top_size();
                MTappend(&pp,-23300);
                MTappend(&pp,num_slice);
                for (int j=0; j<num_slice; j++)
                {
                    MTappend(&pp,-233);
                }
            }
            else
            {
                int num_slice = slice_param.slice_point_size() + 1;
                MTappend(&pp,-23300);
                MTappend(&pp,num_slice);
                int prev_offset = 0;
                for (int j=0; j<slice_param.slice_point_size(); j++)
                {
                    int offset = slice_param.slice_point(j);
                    MTappend(&pp,offset - prev_offset);
                    prev_offset = offset;
                }
                MTappend(&pp,-233);
            }
            int dim = slice_param.axis() - 1;
            MTappend(&pp,1);
            MTappend(&pp,dim);
        }
        else if (layer.type() == "Softmax")
        {
            const caffe::SoftmaxParameter& softmax_param = layer.softmax_param();
            int dim = softmax_param.axis() - 1;
            MTappend(&pp,0);
            MTappend(&pp,dim);
        }
        else if (layer.type() == "Threshold")
        {
            const caffe::ThresholdParameter& threshold_param = layer.threshold_param();
            MTappend(&pp,0);
            MTappend(&pp,(float)threshold_param.threshold());
        }
        MTappend(&pp,-233);

        // add split layer if top reference larger than one
        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = blob_name_decorated[layer.top(0)];
            if (bottom_reference.find(blob_name) != bottom_reference.end())
            {
                int refcount = bottom_reference[blob_name];
                if (refcount > 1)
                {
                    char splitname[256];
                    sprintf(splitname, "splitncnn_%d", internal_split);
                    sprintf(tmp,"%s",blob_name.c_str());
                    MTappend(&pp,(char *)"Split");
                    MTappend(&pp,splitname);
                    MTappend(&pp,1);
                    MTappend(&pp,refcount);
                    MTappend(&pp,tmp);

                    for (int j=0; j<refcount; j++)
                    {
                        sprintf(tmp, "%s_splitncnn_%d", blob_name.c_str(), j);
                        MTappend(&pp,tmp);
                    }
                    MTappend(&pp,-233);

                    internal_split++;
                }
            }
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
                if (bottom_reference.find(blob_name) != bottom_reference.end())
                {
                    int refcount = bottom_reference[blob_name];
                    if (refcount > 1)
                    {
                        char splitname[256];
                        sprintf(splitname, "splitncnn_%d", internal_split);
                        sprintf(tmp,"%s",blob_name.c_str());
                        MTappend(&pp,(char *)"Split");
                        MTappend(&pp,splitname);
                        MTappend(&pp,1);
                        MTappend(&pp,refcount);
                        MTappend(&pp,tmp);

                        for (int j=0; j<refcount; j++)
                        {
                            sprintf(tmp, "%s_splitncnn_%d", blob_name.c_str(), j);
                            MTappend(&pp,tmp);
                        }
                        MTappend(&pp,-233);

                        internal_split++;
                    }
                }
            }
        }

    }

    *ppm = (unsigned char *)pp.buf;
    *bpm = (unsigned char *)bp.buf;
    return 0;

}

}
