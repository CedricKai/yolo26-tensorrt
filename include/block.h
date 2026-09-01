#pragma once
#include <map>
#include <string>
#include <vector>
#include <NvInfer.h>

using namespace nvinfer1;

struct WeightHolder {
    std::map<std::string, Weights> wmap;
    std::vector<float*> bufs;

    ~WeightHolder() {
        for (auto p : bufs) {
            delete[] p;
        }
        bufs.clear();
        wmap.clear();
    }
};

WeightHolder loadWeights(const std::string& file);


ITensor* conv_2d(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int ch, int k=1, int s=1, int p=-1, int g=1, int d=1);


ITensor* conv_2_detect(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int ch, int k=1, int s=1, int p=-1, int g=1, int d=1);


ITensor* batch_norm_2d(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname);

ITensor* conv_bn_silu(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int ch, int k=1, int s=1, int p=-1, int g=1, int d=1, bool act=true);

ITensor* bottleneck(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, bool shortcut=true, int g=1, int k=3, float e=0.5);


ITensor* C3k(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, int n=1, bool shortcut=true, int g=1, float e=0.5);


ITensor* C3k2(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, int n=1, bool shortcut=true, int g=1, float e=0.5, bool atten=false, bool c3k=false);

ITensor* SPPF(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, int k=5, int n=3, bool shortcut=false);

ITensor* attention(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int dim, int num_heads=8, float attn_ratio=0.5);

ITensor* psa_block(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c, float attn_ratio=0.5, int num_heads=4, bool shortcut=true);

ITensor* C2PSA(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, int n=1, float e=0.5);


struct Yolo26Detect {
    ITensor* boxes = nullptr;
    ITensor* scores = nullptr;
};


void build_detect_head(INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    Yolo26Detect& det,
    const std::string& lname1,
    const std::string& lname2,
    int in_c,
    int bs,
    int c2,
    int c3);

IPluginV3Layer* add_yolo_layer(INetworkDefinition* network,
    std::vector<ITensor*> dets,
    std::vector<int>& px_array);