#pragma once
#include "block.h"

struct Yolo26BackboneLayers
{
    ITensor* layer0 = nullptr;
    ITensor* layer1 = nullptr;
    ITensor* layer2 = nullptr;
    ITensor* layer3 = nullptr;    // P3
    ITensor* layer4 = nullptr;
    ITensor* layer5 = nullptr;    // P4
    ITensor* layer6 = nullptr;
    ITensor* layer7 = nullptr;    // P5
    ITensor* layer8 = nullptr;
    ITensor* layer9 = nullptr;    // SPPF
    ITensor* layer10 = nullptr;   // C2PSA
};

void build_yolo26_backbone(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const float& gw,
    const float& gd,
    const int& max_channels,
    const bool& csk,
    const float& w_ratio,
    Yolo26BackboneLayers& backbone);

struct Yolo26NeckLayers
{
    ITensor* layer11 = nullptr;
    ITensor* layer12 = nullptr;
    ITensor* layer13 = nullptr;
    ITensor* layer14 = nullptr;
    ITensor* layer15 = nullptr;
    ITensor* layer16 = nullptr;
    ITensor* layer17 = nullptr;
    ITensor* layer18 = nullptr;
    ITensor* layer19 = nullptr;
    ITensor* layer20 = nullptr;
    ITensor* layer21 = nullptr;
    ITensor* layer22 = nullptr;
};

void build_yolo26_neck(
    INetworkDefinition* network,
    const Yolo26BackboneLayers& bb,
    WeightHolder& wHolder,
    const float& gw,
    const float& gd,
    const int& max_channels,
    const float& w_ratio,
    const float& d_ratio,
    Yolo26NeckLayers& neck);


struct Yolo26DetectLayers {
    ITensor* boxes;
    ITensor* scores;
    std::vector<ITensor*> feats;
};

void build_yolo26_detect(
    INetworkDefinition* network,
    const Yolo26NeckLayers& neck,
    WeightHolder& wHolder,
    std::vector<Yolo26Detect>& dtp);

IHostMemory* build_yolo26_engine(IBuilder* builder,
                                            IBuilderConfig* config,
                                            DataType dt,
                                            const std::string& wts_path,
                                            const float& gd,
                                            const float& gw,
                                            const int& max_channels,
                                            const std::string& model_scale);