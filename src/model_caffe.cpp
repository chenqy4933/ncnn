
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
#include <google/protobuf/message.h>

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
		stringInitBuf(&snew, newlen + Mem64k);
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


bool Model_Caffe::read_proto_from_text(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, message);

    fs.close();

    return success;
}

bool Model_Caffe::read_proto_from_binary(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

int Model_Caffe::caffe2ncnn(const char* caffeproto,
		const char* caffemodel,
		unsigned char** ppm,
		unsigned char** bpm,
        long *model_mem_len)
{
    //current only use quantize_level = 0
    int quantize_level = 0;
    caffe::NetParameter proto;
    caffe::NetParameter net;
    char tmp[4096];

    if (quantize_level != 0 && quantize_level != 256 && quantize_level != 65536) {
        fprintf(stderr, "NCNN: only support quantize level = 0, 256, or 65536");
        return -1;
    }
    // load
    bool s0 = read_proto_from_text(caffeproto, &proto);
    if (!s0)
    {
        fprintf(stderr, "read_proto_from_text failed\n");
        return -1;
    }

    bool s1 = read_proto_from_binary(caffemodel, &net);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    //std::vector<Mat> bp;
    MTString pp;
    stringInitBuf(&pp,Mem64k);
    MTString bp;
    stringInitBuf(&bp,Mem64k);
    int parm_int;
    float parm_float;
    char* parm_str;
    int parm_str_len;
    
    // magic
    //bp.push_back();
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
    
    //sprintf(tmp, "%lu %lu\n", layer_count + bottom_reference.size(), blob_names.size() + splitncnn_blob_count);
    //pp += tmp;
    parm_int = layer_count + bottom_reference.size();
    stringAppend(&pp,&parm_int,sizeof(int));
    parm_int = blob_names.size() + splitncnn_blob_count;
    stringAppend(&pp,&parm_int,sizeof(int));

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
                //sprintf(tmp, "%-16s", "ConvolutionDepthWise");
                //pp += tmp;
                stringAppend(&pp,"ConvolutionDepthWise",strlen("ConvolutionDepthWise"));
                stringAppend(&pp,"\0",1);
            }else{
                //sprintf(tmp, "%-16s", "Convolution");
                //pp += tmp;
                stringAppend(&pp,"Convolution",strlen("Convolution"));
                stringAppend(&pp,"\0",1);
            }
        }
        else if (layer.type() == "ConvolutionDepthwise")
        {
            //sprintf(tmp, "%-16s", "ConvolutionDepthWise");
            //pp += tmp;
            stringAppend(&pp,"ConvolutionDepthWise",strlen("ConvolutionDepthWise"));
            stringAppend(&pp,"\0",1);
        }
        else if (layer.type() == "Deconvolution")
        {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            if (convolution_param.group() != 1){
                //sprintf(tmp, "%-16s", "DeconvolutionDepthWise");
                //pp += tmp;
                stringAppend(&pp,"DeconvolutionDepthWise",strlen("DeconvolutionDepthWise"));
                stringAppend(&pp,"\0",1);
            }else{
                //sprintf(tmp, "%-16s", "Deconvolution");
                //pp += tmp;
                stringAppend(&pp,"Deconvolution",strlen("Deconvolution"));
                stringAppend(&pp,"\0",1);
            }
        }
        else if (layer.type() == "MemoryData")
        {
            //sprintf(tmp, "%-16s", "Input");
            //pp += tmp;
            stringAppend(&pp,"Input",strlen("Input"));
            stringAppend(&pp,"\0",1);
        }
        else if (layer.type() == "Python")
        {
            const caffe::PythonParameter& python_param = layer.python_param();
            std::string python_layer_name = python_param.layer();
            if (python_layer_name == "ProposalLayer"){
                //sprintf(tmp, "%-16s", "Proposal");
                //pp += tmp;
                stringAppend(&pp,"Proposal",strlen("Proposal"));
                stringAppend(&pp,"\0",1);
            }else{
                sprintf(tmp, "%s", python_layer_name.c_str());
                //pp += tmp;
                parm_str = tmp;
                parm_str_len = strlen(parm_str);
                stringAppend(&pp,parm_str,parm_str_len);
                stringAppend(&pp,"\0",1);
            }
        }
        else
        {
            sprintf(tmp, "%s", layer.type().c_str());
            //pp += tmp;
            //parm_int = layer_to_index(layer.type().c_str());
            //stringAppend(&pp,&parm_int,sizeof(int));
            parm_str = tmp;
            parm_str_len = strlen(parm_str);
            stringAppend(&pp,parm_str,parm_str_len);
            stringAppend(&pp,"\0",1);
        }
        //sprintf(tmp, " %-16s %d %d", layer.name().c_str(), layer.bottom_size(), layer.top_size());
        //pp += tmp;
        sprintf(tmp, "%s", layer.name().c_str());
        parm_str = tmp;
        parm_str_len = strlen(parm_str);
        stringAppend(&pp,parm_str,parm_str_len);
        stringAppend(&pp,"\0",1);
        parm_int = layer.bottom_size();
        stringAppend(&pp,&parm_int,sizeof(int));
        parm_int = layer.top_size();
        stringAppend(&pp,&parm_int,sizeof(int));
        
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
            //pp += tmp;
            parm_str = tmp;
            parm_str_len = strlen(parm_str);
            stringAppend(&pp,parm_str,parm_str_len);
            stringAppend(&pp,"\0",1);
        }

        // if input = output, rename: blobname --> blobname_layername
        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = layer.top(0) + "_" + layer.name();
            blob_name_decorated[layer.top(0)] = blob_name;

            sprintf(tmp, "%s", blob_name.c_str());
            //pp += tmp;
            parm_str = tmp;
            parm_str_len = strlen(parm_str);
            stringAppend(&pp,parm_str,parm_str_len);
            stringAppend(&pp,"\0",1);
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
                sprintf(tmp, "%s", blob_name.c_str());
                //pp += tmp;
                parm_str = tmp;
                parm_str_len = strlen(parm_str);
                stringAppend(&pp,parm_str,parm_str_len);
                stringAppend(&pp,"\0",1);
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
            const caffe::BlobProto& var_blob = binlayer.blobs(1);
            //sprintf(tmp, " 0=%d", (int)mean_blob.data_size());
            //pp += tmp;
            parm_int = 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = (int)mean_blob.data_size();
            stringAppend(&pp,&parm_int,sizeof(int));

            const caffe::BatchNormParameter& batch_norm_param = layer.batch_norm_param();
            float eps = batch_norm_param.eps();

            std::vector<float> ones(mean_blob.data_size(), 1.f);
            stringAppend(&bp, ones.data(), sizeof(float) * ones.size());// slope

            if (binlayer.blobs_size() < 3)
            {
                stringAppend(&bp, mean_blob.data().data(), sizeof(float) * mean_blob.data_size());
                float tmp;
                for (int j=0; j<var_blob.data_size(); j++)
                {
                    tmp = var_blob.data().data()[j] + eps;
                    stringAppend(&bp, &tmp, sizeof(float) * 1);
                }
            }
            else
            {
                float scale_factor = 1 / binlayer.blobs(2).data().data()[0];
                // premultiply scale_factor to mean and variance
                float tmp;
                for (int j=0; j<mean_blob.data_size(); j++)
                {
                    tmp = mean_blob.data().data()[j] * scale_factor;
                    stringAppend(&bp, &tmp, sizeof(float) * 1);
                }
                for (int j=0; j<var_blob.data_size(); j++)
                {
                    tmp = var_blob.data().data()[j] * scale_factor + eps;
                    stringAppend(&bp, &tmp, sizeof(float) * 1);
                }
            }

            std::vector<float> zeros(mean_blob.data_size(), 0.f);
            stringAppend(&bp, zeros.data(), sizeof(float) * zeros.size());// bias
        }
        else if (layer.type() == "Concat")
        {
            const caffe::ConcatParameter& concat_param = layer.concat_param();
            int dim = concat_param.axis() - 1;
            //sprintf(tmp, " 0=%d", dim);
            //pp += tmp;
            parm_int = 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = dim;
            stringAppend(&pp,&parm_int,sizeof(int));
        }
        else if (layer.type() == "Convolution" || layer.type() == "ConvolutionDepthwise")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            //sprintf(tmp, " 0=%d", convolution_param.num_output());
            //pp += tmp;
            parm_int = 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = convolution_param.num_output();
            stringAppend(&pp,&parm_int,sizeof(int));
            
            if (convolution_param.has_kernel_w() && convolution_param.has_kernel_h())
            {
                //sprintf(tmp, " 1=%d 11=%d", convolution_param.kernel_w(), convolution_param.kernel_h());
                //pp += tmp;
                parm_int = 1;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.kernel_w();
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 11;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.kernel_h();
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            else
            {
                //sprintf(tmp, " 1=%d", convolution_param.kernel_size(0));
                //pp += tmp;
                parm_int = 1;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.kernel_size(0);
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            //sprintf(tmp, " 2=%d", convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1);
            //pp += tmp;
            parm_int = 2;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1;
            stringAppend(&pp,&parm_int,sizeof(int));
            if (convolution_param.has_stride_w() && convolution_param.has_stride_h())
            {
                //sprintf(tmp, " 3=%d 13=%d", convolution_param.stride_w(), convolution_param.stride_h());
                //pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.stride_w();
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 13;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.stride_h();
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            else
            {
                //sprintf(tmp, " 3=%d", convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1);
                //pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1;
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            /*sprintf(tmp, " 4=%d 5=%d 6=%d", convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0
            , convolution_param.bias_term()
            , weight_blob.data_size());*/
            //pp += tmp;
            parm_int = 4;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 5;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = convolution_param.bias_term();
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 6;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = weight_blob.data_size();
            stringAppend(&pp,&parm_int,sizeof(int));

            if (layer.type() == "ConvolutionDepthwise")
            {
                //sprintf(tmp, " 7=%d", convolution_param.num_output());
                //pp += tmp;
                parm_int = 7;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.num_output();
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            else if (convolution_param.group() != 1)
            {
                //sprintf(tmp, " 7=%d", convolution_param.group());
                //pp += tmp;
                parm_int = 7;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.group();
                stringAppend(&pp,&parm_int,sizeof(int));
            }

            for (int j = 0; j < binlayer.blobs_size(); j++)
            {
                int quantize_tag = 0;
                const caffe::BlobProto& blob = binlayer.blobs(j);

                std::vector<float> quantize_table;
                std::vector<unsigned char> quantize_index;

                std::vector<unsigned short> float16_weights;

                // we will not quantize the bias values
                if (j == 0 && quantize_level != 0)
                {
                    if (quantize_level == 256)
                    {
                    quantize_tag = quantize_weight((float *)blob.data().data(), blob.data_size(), quantize_level, quantize_table, quantize_index);
                    }
                    else if (quantize_level == 65536)
                    {
                    quantize_tag = quantize_weight((float *)blob.data().data(), blob.data_size(), float16_weights);
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
                    stringAppend(&bp, blob.data().data(), sizeof(float) * blob.data_size());
                    
                }
            }

        }
        else if (layer.type() == "Crop")
        {
            const caffe::CropParameter& crop_param = layer.crop_param();
            int num_offset = crop_param.offset_size();
            int woffset = (num_offset == 2) ? crop_param.offset(0) : 0;
            int hoffset = (num_offset == 2) ? crop_param.offset(1) : 0;
            //sprintf(tmp, " 0=%d 1=%d", woffset, hoffset);
            //pp += tmp;
            parm_int = 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = woffset;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 1;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = hoffset;
            stringAppend(&pp,&parm_int,sizeof(int));
        }
        else if (layer.type() == "Deconvolution")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            //sprintf(tmp, " 0=%d", convolution_param.num_output());
            //pp += tmp;
            parm_int = 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = convolution_param.num_output();
            stringAppend(&pp,&parm_int,sizeof(int));
            if (convolution_param.has_kernel_w() && convolution_param.has_kernel_h())
            {
                //sprintf(tmp, " 1=%d 11=%d", convolution_param.kernel_w(), convolution_param.kernel_h());
                //pp += tmp;
                parm_int = 1;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.kernel_w();
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 11;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.kernel_h();
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            else
            {
                //sprintf(tmp, " 1=%d", convolution_param.kernel_size(0));
                //pp += tmp;
                parm_int = 1;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.kernel_size(0);
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            //sprintf(tmp, " 2=%d", convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1);
            //pp += tmp;
            parm_int = 2;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1;
            stringAppend(&pp,&parm_int,sizeof(int));
            if (convolution_param.has_stride_w() && convolution_param.has_stride_h())
            {
                //sprintf(tmp, " 3=%d 13=%d", convolution_param.stride_w(), convolution_param.stride_h());
                //pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.stride_w();
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 13;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.stride_h();
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            else
            {
                //sprintf(tmp, " 3=%d", convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1);
                //pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1;
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            /*sprintf(tmp, " 4=%d 5=%d 6=%d", convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0
                , convolution_param.bias_term()
                , weight_blob.data_size());*/
            //pp += tmp;
            parm_int = 4;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 5;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = convolution_param.bias_term();
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 6;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = weight_blob.data_size();
            stringAppend(&pp,&parm_int,sizeof(int));

            if (convolution_param.group() != 1)
            {
                //sprintf(tmp, " 7=%d", convolution_param.group());
                //pp += tmp;
                parm_int = 7;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = convolution_param.group();
                stringAppend(&pp,&parm_int,sizeof(int));
            }

            int quantized_weight = 0;
            stringAppend(&bp, &quantized_weight, sizeof(int)* 1);
            
            // reorder weight from inch-outch to outch-inch
            int ksize = convolution_param.kernel_size(0);
            int num_output = convolution_param.num_output();
            int num_input = weight_blob.data_size() / (ksize * ksize) / num_output;
            const float* weight_data_ptr = weight_blob.data().data();
            for (int k=0; k<num_output; k++)
            {
                for (int j=0; j<num_input; j++)
                {
                    stringAppend(&bp, weight_data_ptr + (j*num_output + k) * ksize * ksize, sizeof(float) * ksize * ksize);
                }
            }

            for (int j=1; j<binlayer.blobs_size(); j++)
            {
                const caffe::BlobProto& blob = binlayer.blobs(j);
                stringAppend(&bp, blob.data().data(), sizeof(float) * blob.data_size());
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
            //sprintf(tmp, " 0=%d -23301=%d", (int)eltwise_param.operation(), coeff_size);
            //pp += tmp;
            parm_int = 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = (int)eltwise_param.operation();
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = -23301;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = coeff_size;
            stringAppend(&pp,&parm_int,sizeof(int));
            for (int j=0; j<coeff_size; j++)
            {
                //sprintf(tmp, ",%f", eltwise_param.coeff(j));
                //pp += tmp;
                parm_float = eltwise_param.coeff(j);
                stringAppend(&pp,&parm_float,sizeof(float));
            }
        }
        else if (layer.type() == "ELU")
        {
            const caffe::ELUParameter& elu_param = layer.elu_param();
            //sprintf(tmp, " 0=%f", elu_param.alpha());
            //pp += tmp;
            parm_int = 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_float = elu_param.alpha();
            stringAppend(&pp,&parm_float,sizeof(float));
        }
        else if (layer.type() == "InnerProduct")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::InnerProductParameter& inner_product_param = layer.inner_product_param();
           /* sprintf(tmp, " 0=%d 1=%d 2=%d", inner_product_param.num_output()
                , inner_product_param.bias_term()
                , weight_blob.data_size());
            pp += tmp;*/
            parm_int = 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = inner_product_param.num_output();
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 1;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = inner_product_param.bias_term();
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 2;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = weight_blob.data_size();
            stringAppend(&pp,&parm_int,sizeof(int));

            for (int j=0; j<binlayer.blobs_size(); j++)
            {
                int quantize_tag = 0;
                const caffe::BlobProto& blob = binlayer.blobs(j);

                std::vector<float> quantize_table;
                std::vector<unsigned char> quantize_index;

                std::vector<unsigned short> float16_weights;

                // we will not quantize the bias values
                if (j == 0 && quantize_level != 0)
                {
                    if (quantize_level == 256)
                    {
                    quantize_tag = quantize_weight((float *)blob.data().data(), blob.data_size(), quantize_level, quantize_table, quantize_index);
                    }
                    else if (quantize_level == 65536)
                    {
                    quantize_tag = quantize_weight((float *)blob.data().data(), blob.data_size(), float16_weights);
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
                    stringAppend(&bp, blob.data().data(), sizeof(float) * blob.data_size());
                }
            }
        }
        else if (layer.type() == "Input")
        {
            const caffe::InputParameter& input_param = layer.input_param();
            const caffe::BlobShape& bs = input_param.shape(0);
            if (bs.dim_size() == 4)
            {
                /*sprintf(tmp, " 0=%ld 1=%ld 2=%ld", bs.dim(3)
                    , bs.dim(2)
                    , bs.dim(1));
                pp += tmp;*/
                parm_int = 0;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = bs.dim(3);
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 1;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = bs.dim(2);
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 2;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = bs.dim(1);
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            else if (bs.dim_size() == 3)
            {
                /*sprintf(tmp, " 0=%ld 1=%ld 2=-233", bs.dim(2)
                    , bs.dim(1));
                pp += tmp;*/
                parm_int = 0;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = bs.dim(2);
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 1;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = bs.dim(1);
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 2;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = -233;
                stringAppend(&pp,&parm_int,sizeof(int));
            }
            else if (bs.dim_size() == 2)
            {
                /*sprintf(tmp, " 0=%ld 1=-233 2=-233", bs.dim(1));
                pp += tmp;*/
                parm_int = 0;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = bs.dim(1);
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 1;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = -233;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 2;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = -233;
                stringAppend(&pp,&parm_int,sizeof(int));
            }
        }
        else if (layer.type() == "Interp")
        {
            const caffe::InterpParameter& interp_param = layer.interp_param();
            /*sprintf(tmp, " 0=%d 1=%f 2=%f 3=%d 4=%d", 2
                , (float)interp_param.zoom_factor()
                , (float)interp_param.zoom_factor()
                , interp_param.height()
                , interp_param.width());
            pp += tmp;*/
            parm_int = 0;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 2;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 1;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_float = (float)interp_param.zoom_factor();
            stringAppend(&pp,&parm_float,sizeof(parm_float));
            parm_int = 2;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_float = (float)interp_param.zoom_factor();
            stringAppend(&pp,&parm_float,sizeof(parm_float));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = interp_param.height();
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 4;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = interp_param.width();
            stringAppend(&pp,&parm_int,sizeof(int));
        }
        else if (layer.type() == "LRN")
        {
            const caffe::LRNParameter& lrn_param = layer.lrn_param();
            /*sprintf(tmp, " 0=%d 1=%d 2=%f 3=%f", lrn_param.norm_region()
                , lrn_param.local_size()
                , lrn_param.alpha()
                , lrn_param.beta());
            pp += tmp;*/
            MTappend(&pp,0);
            parm_int = lrn_param.norm_region();
            MTappend(&pp,parm_int);
            MTappend(&pp,1);
            MTappend(&pp,(int)lrn_param.local_size());
            MTappend(&pp,2);
            MTappend(&pp,(float)lrn_param.alpha());
            MTappend(&pp,3);
            MTappend(&pp,(float)lrn_param.beta());
            /*parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
        }
        else if (layer.type() == "MemoryData")
        {
            const caffe::MemoryDataParameter& memory_data_param = layer.memory_data_param();
            /*sprintf(tmp, " 0=%d 1=%d 2=%d", memory_data_param.width()
                , memory_data_param.height()
                , memory_data_param.channels());
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
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
            /*sprintf(tmp, " 0=%d 1=%d 2=%f", mvn_param.normalize_variance()
                , mvn_param.across_channels()
                , mvn_param.eps());
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            MTappend(&pp,0);
            MTappend(&pp,mvn_param.normalize_variance());
            MTappend(&pp,1);
            MTappend(&pp,mvn_param.across_channels());
            MTappend(&pp,2);
            MTappend(&pp,mvn_param.eps());
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
            /*sprintf(tmp, " 0=%d", pooling_param.pool());
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            MTappend(&pp,0);
            MTappend(&pp,pooling_param.pool()); 
            if (pooling_param.has_kernel_w() && pooling_param.has_kernel_h())
            {
                /*sprintf(tmp, " 1=%d", pooling_param.kernel_w());
                sprintf(tmp, " 11=%d", pooling_param.kernel_h());
                pp += tmp;
                parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,1);
                MTappend(&pp,(int)pooling_param.kernel_w());
                MTappend(&pp,11);
                MTappend(&pp,(int)pooling_param.kernel_h());
            }
            else
            {
                /*sprintf(tmp, " 1=%d", pooling_param.kernel_size());
                pp += tmp;
                parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,1);
                MTappend(&pp,(int)pooling_param.kernel_size());
            }
            if (pooling_param.has_stride_w() && pooling_param.has_stride_h())
            {
                /*sprintf(tmp, " 2=%d 12=%d", pooling_param.stride_w()
                    , pooling_param.stride_h());
                pp += tmp;
                parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,2);
                MTappend(&pp,(int)pooling_param.stride_w());
                MTappend(&pp,12);
                MTappend(&pp,(int)pooling_param.stride_h());
            }
            else
            {
                /*sprintf(tmp, " 2=%d", pooling_param.stride());
                pp += tmp;
                parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,2);
                MTappend(&pp,(int)pooling_param.stride());
            }
            if (pooling_param.has_pad_w() && pooling_param.has_pad_h())
            {
                /*sprintf(tmp, " 3=%d 13=%d", pooling_param.pad_w()
                    , pooling_param.pad_h());
                pp += tmp;
                parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,3);
                MTappend(&pp,(int)pooling_param.pad_w());
                MTappend(&pp,13);
                MTappend(&pp,(int)pooling_param.pad_h());
            }
            else
            {
                /*sprintf(tmp, " 3=%d", pooling_param.pad());
                pp += tmp;
                parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,3);
                MTappend(&pp,(int)pooling_param.pad());
            }
            /*sprintf(tmp, " 4=%d", pooling_param.has_global_pooling() ? pooling_param.global_pooling() : 0);
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            MTappend(&pp,4);
            MTappend(&pp,pooling_param.has_global_pooling() ? pooling_param.global_pooling() : 0);
        }
        else if (layer.type() == "Power")
        {
            const caffe::PowerParameter& power_param = layer.power_param();
            /*sprintf(tmp, " 0=%f 1=%f 2=%f", power_param.power()
                , power_param.scale()
                , power_param.shift());
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            MTappend(&pp,0);
            MTappend(&pp,power_param.power());
            MTappend(&pp,1);
            MTappend(&pp,power_param.scale());
            MTappend(&pp,2);
            MTappend(&pp, power_param.shift());
                
        }
        else if (layer.type() == "PReLU")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);
            const caffe::BlobProto& slope_blob = binlayer.blobs(0);
            /*sprintf(tmp, " 0=%d", slope_blob.data_size());
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            MTappend(&pp,0);
            MTappend(&pp, slope_blob.data_size());
            stringAppend(&bp, slope_blob.data().data(), sizeof(float) * slope_blob.data_size());
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
                /*sprintf(tmp, " 0=%d 1=%d 2=%d 3=%d 4=%f 5=%d", feat_stride
                    , base_size
                    , pre_nms_topN
                    , after_nms_topN
                    , nms_thresh
                    , min_size);
                pp += tmp;
                parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
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
                //sprintf(tmp, " 0=%f", relu_param.negative_slope());
                //pp += tmp;
                MTappend(&pp,0);
                MTappend(&pp,relu_param.negative_slope());
            }
        }
        else if (layer.type() == "Reshape")
        {
            const caffe::ReshapeParameter& reshape_param = layer.reshape_param();
            const caffe::BlobShape& bs = reshape_param.shape();
            if (bs.dim_size() == 1)
            {
                /*sprintf(tmp, " 0=%ld 1=-233 2=-233", bs.dim(0));
                pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(0));
                MTappend(&pp,1);
                MTappend(&pp,-233);
                MTappend(&pp,2);
                MTappend(&pp,-233);
            }
            else if (bs.dim_size() == 2)
            {
                /*sprintf(tmp, " 0=%ld 1=%ld 2=-233", bs.dim(1), bs.dim(0));
                pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(1));
                MTappend(&pp,1);
                MTappend(&pp,(int)bs.dim(0));
                MTappend(&pp,2);
                MTappend(&pp,-233);
            }
            else if (bs.dim_size() == 3)
            {
                /*sprintf(tmp, " 0=%ld 1=%ld 2=%ld", bs.dim(2), bs.dim(1), bs.dim(0));
                pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(2));
                MTappend(&pp,1);
                MTappend(&pp,(int)bs.dim(1));
                MTappend(&pp,2);
                MTappend(&pp,(int)bs.dim(0));
            }
            else // bs.dim_size() == 4
            {
                /*sprintf(tmp, " 0=%ld 1=%ld 2=%ld", bs.dim(3), bs.dim(2), bs.dim(1));
                pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,0);
                MTappend(&pp,(int)bs.dim(3));
                MTappend(&pp,1);
                MTappend(&pp,(int)bs.dim(2));
                MTappend(&pp,2);
                MTappend(&pp,(int)bs.dim(1));
            }
            /*sprintf(tmp, " 3=0");// permute
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
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
                /*sprintf(tmp, " 0=%d", (int)weight_blob.data_size());
                pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,0);
                MTappend(&pp,(int)weight_blob.data_size());
            }
            else
            {
                /*sprintf(tmp, " 0=-233");
                pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,0);
                MTappend(&pp,-233);
            }

            /*sprintf(tmp, " 1=%d", scale_param.bias_term());
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            MTappend(&pp,1);
            MTappend(&pp,scale_param.bias_term());

            for (int j=0; j<binlayer.blobs_size(); j++)
            {
                const caffe::BlobProto& blob = binlayer.blobs(j);
                stringAppend(&bp, blob.data().data(), sizeof(float) * blob.data_size());
            }
        }
        else if (layer.type() == "ShuffleChannel")
        {
            const caffe::ShuffleChannelParameter&
                    shuffle_channel_param = layer.shuffle_channel_param();
            /*sprintf(tmp, " 0=%d", shuffle_channel_param.group());
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            MTappend(&pp,0);
            MTappend(&pp,(int)shuffle_channel_param.group());
        }
        else if (layer.type() == "Slice")
        {
            const caffe::SliceParameter& slice_param = layer.slice_param();
            if (slice_param.has_slice_dim())
            {
                int num_slice = layer.top_size();
                /*sprintf(tmp, " -23300=%d", num_slice);
                pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,-23300);
                MTappend(&pp,num_slice);
                for (int j=0; j<num_slice; j++)
                {
                    /*sprintf(tmp, ",-233");
                    pp += tmp;
                    parm_int = 3;
                    stringAppend(&pp,&parm_int,sizeof(int));
                    parm_int = ;
                    stringAppend(&pp,&parm_int,sizeof(int));*/
                    MTappend(&pp,-233);
                }
            }
            else
            {
                int num_slice = slice_param.slice_point_size() + 1;
                /*sprintf(tmp, " -23300=%d", num_slice);
                pp += tmp;
                parm_int = 3;
                stringAppend(&pp,&parm_int,sizeof(int));
                parm_int = ;
                stringAppend(&pp,&parm_int,sizeof(int));*/
                MTappend(&pp,-23300);
                MTappend(&pp,num_slice);
                int prev_offset = 0;
                for (int j=0; j<slice_param.slice_point_size(); j++)
                {
                    int offset = slice_param.slice_point(j);
                    //sprintf(tmp, ",%d", offset - prev_offset);
                    //pp += tmp;
                    MTappend(&pp,offset - prev_offset);
                    prev_offset = offset;
                }
                //sprintf(tmp, ",-233");
                //pp += tmp;
                MTappend(&pp,-233);
            }
            int dim = slice_param.axis() - 1;
            /*sprintf(tmp, " 1=%d", dim);
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            MTappend(&pp,1);
            MTappend(&pp,dim);
        }
        else if (layer.type() == "Softmax")
        {
            const caffe::SoftmaxParameter& softmax_param = layer.softmax_param();
            int dim = softmax_param.axis() - 1;
            /*sprintf(tmp, " 0=%d", dim);
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            MTappend(&pp,0);
            MTappend(&pp,dim);
        }
        else if (layer.type() == "Threshold")
        {
            const caffe::ThresholdParameter& threshold_param = layer.threshold_param();
            /*sprintf(tmp, " 0=%f", threshold_param.threshold());
            pp += tmp;
            parm_int = 3;
            stringAppend(&pp,&parm_int,sizeof(int));
            parm_int = ;
            stringAppend(&pp,&parm_int,sizeof(int));*/
            parm_float = threshold_param.threshold();
            MTappend(&pp,0);
            MTappend(&pp,parm_float);
        }
        MTappend(&pp,-233);

        //sprintf(tmp, "\n");
        //pp += tmp;

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
                    /*sprintf(tmp, "%-16s %-16s %d %d  %s", "Split", splitname, 1, refcount
                        , blob_name.c_str());
                    pp += tmp;*/
                    sprintf(tmp,"%s",blob_name.c_str());
                    MTappend(&pp,"Split");
                    MTappend(&pp,splitname);
                    MTappend(&pp,1);
                    MTappend(&pp,refcount);
                    MTappend(&pp,tmp);

                    for (int j=0; j<refcount; j++)
                    {
                        sprintf(tmp, "%s_splitncnn_%d", blob_name.c_str(), j);
                        //pp += tmp;
                        MTappend(&pp,tmp);
                    }
                    //sprintf(tmp, "\n");
                    //pp += tmp;
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
                        /*sprintf(tmp, "%-16s %-16s %d %d %s", "Split", splitname, 1, refcount
                            , blob_name.c_str());
                        pp += tmp;*/
                        sprintf(tmp,"%s",blob_name.c_str());
                        MTappend(&pp,"Split");
                        MTappend(&pp,splitname);
                        MTappend(&pp,1);
                        MTappend(&pp,refcount);
                        MTappend(&pp,tmp);

                        for (int j=0; j<refcount; j++)
                        {
                            sprintf(tmp, "%s_splitncnn_%d", blob_name.c_str(), j);
                            //pp += tmp;
                            MTappend(&pp,tmp);
                        }
                        //sprintf(tmp, "\n");
                        //pp += tmp;
                        MTappend(&pp,-233);

                        internal_split++;
                    }
                }
            }
        }

    }

    *ppm = (unsigned char *)pp.buf;
    *bpm = (unsigned char *)bp.buf;
    *model_mem_len = bp.len;
    return 0;

}

}
