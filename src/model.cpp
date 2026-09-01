#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "model.h"
#include "config.h"


static void check_weights_nan_inf(const char* layerName, const char* weightName, Weights w)    // NOLINT
{
    if (w.values == nullptr || w.count <= 0) {
        //printf("  !!! ALERT !!! Layer[%s] %s contains nullptr or 0 !!!\n", layerName, weightName);
        return;
    }

    if (w.type == DataType::kFLOAT)
    {
        auto* pData = static_cast<const float*>(w.values);
        bool hasNaN = false;
        bool hasInf = false;
        float min_v = std::numeric_limits<float>::max();
        float max_v = -std::numeric_limits<float>::max();

        for (int64_t k = 0; k < w.count; k++)
        {
            float val = pData[k];
            if (std::isnan(val))
            {
                hasNaN = true;
            }
            if (std::isinf(val))
            {
                hasInf = true;
            }
            if (val < min_v) min_v = val;
            if (val > max_v) max_v = val;
        }

        printf("     %s range: [%.6f, %.6f], hasNaN=%d, hasInf=%d\n",
               weightName, min_v, max_v, hasNaN, hasInf);

        if (hasNaN || hasInf)
        {
            printf("  !!! ALERT !!! Layer[%s] %s contains NaN/Inf !!!\n", layerName, weightName);
        }
    }

}

void print_network_layers(const INetworkDefinition* network)
{
    printf("===== Network Layer Info =====\n");
    int numLayers = network->getNbLayers();
    for (int i = 0; i < numLayers; i++)
    {
        auto layer = network->getLayer(i);
        LayerType layer_type = layer->getType();
        const char* layerName = layer->getName();
        printf("[Layer %d] Type=%d(), Name=%s\n", i, static_cast<int>(layer_type), layerName);

        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            auto tensor = layer->getInput(j);
            Dims dims = tensor->getDimensions();
            printf("  Input[%d]: %s, dims(", j, tensor->getName());
            for (int d = 0; d < dims.nbDims; d++)
            {
                printf("%ld ", dims.d[d]);
            }
            printf(")\n");
        }

        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            auto tensor = layer->getOutput(j);
            Dims dims = tensor->getDimensions();
            printf("  Output[%d]: %s, dims(", j, tensor->getName());
            for (int d = 0; d < dims.nbDims; d++)
            {
                printf("%ld ", dims.d[d]);
            }
            printf(")\n");
        }

        if (auto convLayer = dynamic_cast<IConvolutionLayer*>(layer))
        {
            Weights kernelW = convLayer->getKernelWeights();
            Weights biasW   = convLayer->getBiasWeights();
            printf("  -> Conv Kernel Weights: count=%ld, dtype=%d\n", kernelW.count, static_cast<int>(kernelW.type));
            check_weights_nan_inf(layerName, "Kernel", kernelW);

            printf("  -> Conv Bias Weights:   count=%ld, dtype=%d\n", biasW.count, static_cast<int>(biasW.type));
            check_weights_nan_inf(layerName, "Bias", biasW);
        }
        else if (auto scaleLayer = dynamic_cast<IScaleLayer*>(layer))
        {
            Weights shiftW = scaleLayer->getShift();
            Weights scaleW = scaleLayer->getScale();
            Weights powerW = scaleLayer->getPower();
            printf("  -> Scale Shift count=%ld, dtype=%d\n", shiftW.count, static_cast<int>(shiftW.type));
            check_weights_nan_inf(layerName, "Shift", shiftW);

            printf("  -> Scale Scale count=%ld, dtype=%d\n", scaleW.count, static_cast<int>(scaleW.type));
            check_weights_nan_inf(layerName, "Scale", scaleW);

            printf("  -> Scale Power count=%ld, dtype=%d\n", powerW.count, static_cast<int>(powerW.type));
            check_weights_nan_inf(layerName, "Power", powerW);
        }
        printf("\n");
    }
}

static int get_width(int base_ch, float gw, int max_channels, int divisor = 8) {    // NOLINT
    float limited = std::min(base_ch, max_channels) * gw;                       // NOLINT
    int channel = static_cast<int>((limited + divisor / 2.0) / divisor) * divisor;
    return std::max(channel, divisor);
}


static int get_depth(int x, float gd) {                         // NOLINT
    if (x == 1)                                                 // NOLINT
        return 1;                                               // NOLINT
    int r = round(x * gd);                                      // NOLINT
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0)  // NOLINT
        --r;
    return std::max<int>(r, 1);
}


static void calculate_strides(const std::vector<ITensor*>& dts, int reference_size, std::vector<int>& ss) {
    for (int i = 0; i < dts.size(); i++) {
        Dims dims = dts[i]->getDimensions();
        ss[i] = reference_size / static_cast<int>(dims.d[2]);
    }
}

void build_yolo26_backbone(
        INetworkDefinition* network,
        ITensor& input,
        WeightHolder& wHolder,
        const float& gw,
        const float& gd,
        const int& max_channels,
        const bool& csk,
        const float& w_ratio,
        Yolo26BackboneLayers& bb) {

        bb.layer0 = conv_bn_silu(network, input, wHolder, "model.0",
            get_width(64, gw, max_channels), 3, 2);

        bb.layer1 = conv_bn_silu(network, *bb.layer0, wHolder, "model.1",
            get_width(128, gw, max_channels), 3, 2);

        bb.layer2 = C3k2(network, *bb.layer1, wHolder, "model.2",
            get_width(256, gw, max_channels), get_width(256, gw, max_channels),
            get_depth(2, gd), true, 1, 0.25, false, csk);

        bb.layer3 = conv_bn_silu(network, *bb.layer2, wHolder, "model.3",
            get_width(256, gw, max_channels), 3, 2, 1);

        // P3
        bb.layer4 = C3k2(network, *bb.layer3, wHolder, "model.4",
        get_width(512, gw, max_channels), get_width(512, gw, max_channels),
        get_depth(2, gd), true, 1, 0.25, false, csk);

        bb.layer5 = conv_bn_silu(network, *bb.layer4, wHolder, "model.5",
            get_width(512, gw, max_channels), 3, 2, 1);

        // P4
        bb.layer6 = C3k2(network, *bb.layer5, wHolder, "model.6",
        get_width(512, gw, max_channels), get_width(512, gw, max_channels),
        get_depth(2, gd), true, 1, 0.5, false, true);

        bb.layer7 = conv_bn_silu(network, *bb.layer6, wHolder, "model.7",
        get_width(512, gw, max_channels) * w_ratio, 3, 2, 1);

        // P5
        bb.layer8 = C3k2(network, *bb.layer7, wHolder, "model.8",
            get_width(512, gw, max_channels) * w_ratio, get_width(512, gw, max_channels) * w_ratio,
            get_depth(2, gd), true, 1, 0.5, false, true);

        bb.layer9 = SPPF(network, *bb.layer8, wHolder, "model.9",
            get_width(512, gw, max_channels)* w_ratio, get_width(512, gw, max_channels)* w_ratio,
            5, 3, true);

        bb.layer10 = C2PSA(network, *bb.layer9, wHolder, "model.10",
            get_width(512, gw, max_channels)* w_ratio, get_width(512, gw, max_channels)* w_ratio);
}

void build_yolo26_neck(
    INetworkDefinition* network,
    const Yolo26BackboneLayers& bb,
    WeightHolder& wHolder,
    const float& gw,
    const float& gd,
    const int& max_channels,
    const float& ratio,
    const float& d_ratio,
    Yolo26NeckLayers& nk) {
    
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    IResizeLayer* upsample11 = network->addResize(*bb.layer10);
    upsample11->setName("Upsample 11");

    upsample11->setResizeMode(InterpolationMode::kNEAREST);
    upsample11->setScales(scale, 4);
    nk.layer11 = upsample11->getOutput(0);

    ITensor* inputTensor12[] = {nk.layer11, bb.layer6};
    IConcatenationLayer* cat_12 = network->addConcatenation(inputTensor12, 2);
    cat_12->setAxis(1);
    cat_12->setName("Concat 12");
    nk.layer12 = cat_12->getOutput(0);

    nk.layer13 = C3k2(network, *nk.layer12, wHolder, "model.13",
        get_width(512, gw, max_channels), get_width(512, gw, max_channels),
        get_depth(2, gd), true, 1, 0.5, false, true);

    IResizeLayer* upsample14 = network->addResize(*nk.layer13);
    upsample14->setName("Upsample 14");

    upsample14->setResizeMode(InterpolationMode::kNEAREST);
    upsample14->setScales(scale, 4);
    nk.layer14 = upsample14->getOutput(0);

    ITensor* inputTensor15[] = {nk.layer14, bb.layer4};
    IConcatenationLayer* cat_15 = network->addConcatenation(inputTensor15, 2);
    cat_15->setAxis(1);
    cat_15->setName("Concat 15");
    nk.layer15 = cat_15->getOutput(0);

    // P3
    nk.layer16 = C3k2(network, *nk.layer15, wHolder, "model.16",
        get_width(256, gw, max_channels), get_width(256, gw, max_channels),
        get_depth(2, gd), true, 1, 0.5, false, true);

    nk.layer17 = conv_bn_silu(network, *nk.layer16, wHolder, "model.17",
            get_width(256, gw, max_channels), 3, 2);

    ITensor* inputTensor18[] = {nk.layer17, nk.layer13};
    IConcatenationLayer* cat_18 = network->addConcatenation(inputTensor18, 2);
    cat_18->setAxis(1);
    cat_18->setName("Concat 18");
    nk.layer18 = cat_18->getOutput(0);

    // P4
    nk.layer19 = C3k2(network, *nk.layer18, wHolder, "model.19",
        get_width(512, gw, max_channels), get_width(512, gw, max_channels),
        get_depth(2, gd), true, 1, 0.5, false, true);

    nk.layer20 = conv_bn_silu(network, *nk.layer19, wHolder, "model.20",
            get_width(512, gw, max_channels), 3, 2, 1);

    ITensor* inputTensor21[] = {nk.layer20, bb.layer10};
    IConcatenationLayer* cat_21 = network->addConcatenation(inputTensor21, 2);
    cat_21->setAxis(1);
    cat_21->setName("Concat 21");
    nk.layer21 = cat_21->getOutput(0);

    // P5
    nk.layer22 = C3k2(network, *nk.layer21, wHolder, "model.22",
        get_width(512, gw, max_channels) * ratio, get_width(512, gw, max_channels) * ratio,
        get_depth(2, gd) * d_ratio, true, 1, 0.5, true, true);

}


void build_yolo26_detect(
    INetworkDefinition* network,
    const Yolo26NeckLayers& neck,
    WeightHolder& wHolder,
    std::vector<Yolo26Detect>& dtp) {
    
    int ch0 = static_cast<int>(neck.layer16->getDimensions().d[1]);   // ch0
    int ch1 = static_cast<int>(neck.layer19->getDimensions().d[1]);   // ch1
    int ch2 = static_cast<int>(neck.layer22->getDimensions().d[1]);   // ch2
    int c2 = std::max(16, std::max(ch0 / 4, kRegMax * 4));
    int c3 = std::max(ch0, std::min(kNumClass, 100));
    int bs = static_cast<int>(neck.layer16->getDimensions().d[0]);

    // P3
    build_detect_head(network, *neck.layer16, wHolder, dtp[0], "model.23.one2one_cv2.0","model.23.one2one_cv3.0",
        ch0, bs, c2, c3);

    // P4
    build_detect_head(network, *neck.layer19, wHolder, dtp[1], "model.23.one2one_cv2.1","model.23.one2one_cv3.1",
        ch1, bs, c2, c3);

    // P5
    build_detect_head(network, *neck.layer22, wHolder, dtp[2], "model.23.one2one_cv2.2","model.23.one2one_cv3.2",
        ch2, bs, c2, c3);

}

static std::pair<ITensor*, ITensor*> make_anchors(INetworkDefinition* network,
    const std::vector<ITensor*>& feats,
    const std::vector<int>& strides,
    float grid_cell_offset = 0.5f)
{
    std::vector<float> anchor_cx;
    std::vector<float> anchor_cy;
    std::vector<float> stride_data;

    for (size_t i = 0; i < feats.size(); i++)
    {
        auto* feat = feats[i];
        int stride = strides[i];

        // feat shape [B,C,H,W]
        auto dims = feat->getDimensions();
        int64_t H = dims.d[2];
        int64_t W = dims.d[3];

        // sy: 0~H‑1 + offset; sx:0~W‑1 + offset
        for (int y = 0; y < H; y++)
        {
            float cy = y + grid_cell_offset;        // NOLINT
            for (int x = 0; x < W; x++)
            {
                float cx = x + grid_cell_offset;    // NOLINT
                anchor_cx.push_back(cx);
                anchor_cy.push_back(cy);
                stride_data.push_back(static_cast<float>(stride));
            }
        }
    }

    int total_anchors = static_cast<int>(anchor_cx.size());

    // anchor_points_x shape [total_anchors, 1]
    Dims dims_anchors_x{2, {1, total_anchors}};
    Weights w_anchors_x{DataType::kFLOAT, anchor_cx.data(), static_cast<int64_t>(anchor_cx.size())};
    auto* anchor_points_x = network->addConstant(dims_anchors_x, w_anchors_x);
    anchor_points_x->setName("anchor_points_x");

    // anchor_points_y shape [total_anchors, 1]
    Dims dims_anchors_y{2, {1, total_anchors}};
    Weights w_anchors_y{DataType::kFLOAT, anchor_cy.data(), static_cast<int64_t>(anchor_cy.size())};
    auto* anchor_points_y = network->addConstant(dims_anchors_y, w_anchors_y);
    anchor_points_y->setName("anchor_points_y");

    // anchor_points shape [total_anchors, 2]
    ITensor* concat_tensors[] = {anchor_points_x->getOutput(0), anchor_points_y->getOutput(0)};
    auto* concat_layer = network->addConcatenation(concat_tensors, 2);
    concat_layer->setAxis(0);
    concat_layer->setName("anchor_points");

    // stride_tensor shape [total_anchors, 1]
    Dims dims_stride{2, {1, total_anchors}};
    Weights w_stride{DataType::kFLOAT, stride_data.data(), static_cast<int64_t>(stride_data.size())};
    auto* stride_tensor = network->addConstant(dims_stride, w_stride);
    stride_tensor->setName("stride_tensor");

    return {concat_layer->getOutput(0), stride_tensor->getOutput(0)};
}

static void concat_detect_outputs(
    INetworkDefinition* network,
    Yolo26DetectLayers& detect,
    const std::vector<Yolo26Detect>& detect_p,
    const Yolo26NeckLayers& neck) {

    // concat output from [p3, p4, p5]
    ITensor* inputTensor_b[] = {
        detect_p[0].boxes,
        detect_p[1].boxes,
        detect_p[2].boxes,
    };
    IConcatenationLayer* cat_b = network->addConcatenation(inputTensor_b, 3);
    cat_b->setAxis(2);
    cat_b->setName("Cat Boxes");
    detect.boxes = cat_b->getOutput(0);

    // scores for all
    ITensor* inputTensor_s[] = {
        detect_p[0].scores,
        detect_p[1].scores,
        detect_p[2].scores,
    };
    IConcatenationLayer* cat_s = network->addConcatenation(inputTensor_s, 3);
    cat_s->setAxis(2);
    cat_s->setName("Cat Scores");
    detect.scores = cat_s->getOutput(0);

    // feats
    detect.feats = {neck.layer16, neck.layer19, neck.layer22};
}

IHostMemory* build_yolo26_engine(
    IBuilder* builder,
    IBuilderConfig* config,
    DataType dt,
    const std::string& wts_path,
    const float& gd,
    const float& gw,
    const int& max_channels,
    const std::string& model_scale) {

    WeightHolder weightMap = loadWeights(wts_path);
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // adaptive model parameters
    bool csk = true;
    float w_ratio = 1.0;
    float d_ratio = 1.0;
    if (model_scale == "n" || model_scale == "s") {
        csk = false;
        w_ratio = 2.0;
    }
    if (model_scale == "l" || model_scale == "x") {
        d_ratio = 0.5;
    }

    // input
    ITensor* data = network->addInput(kInputTensorName, dt, Dims4{1, 3, kInputH, kInputW});

    // backbone
    Yolo26BackboneLayers backbone;
    build_yolo26_backbone(network, *data, weightMap, gw, gd, max_channels, csk, w_ratio, backbone);

    // neck
    Yolo26NeckLayers neck;
    build_yolo26_neck(network, backbone, weightMap, gw, gd, max_channels, w_ratio, d_ratio, neck);

    // head
    std::vector<Yolo26Detect> detect_p(3);
    build_yolo26_detect(network, neck, weightMap, detect_p);

    // concat detect outputs
    Yolo26DetectLayers detect;
    concat_detect_outputs(network, detect, detect_p, neck);

    // decode
    //pair<ITensor*, ITensor*> anchor_points, stride_tensor = makeAnchors(network, detect.feats, {8,16,32}, 0.5f);

    // strides
    std::vector<int> strides(detect.feats.size());
    calculate_strides(detect.feats, kInputH, strides);

    //_inference plugin
    // concat (decode dbox) and (sigmoid scores)
    IActivationLayer* sigmoid_layer = network->addActivation(*detect.scores, ActivationType::kSIGMOID);
    sigmoid_layer->setName("sigmoid_scores");
    IShuffleLayer* trans_scores = network->addShuffle(*sigmoid_layer->getOutput(0));
    trans_scores->setName("Trans Scores");
    trans_scores->setFirstTranspose(Permutation{0,2,1});

    IShuffleLayer* trans_boxes = network->addShuffle(*detect.boxes);
    trans_boxes->setName("Trans Boxes");
    trans_boxes->setFirstTranspose(Permutation{0,2,1});

    IPluginV3Layer* yolo = add_yolo_layer(network,std::vector<ITensor*>{trans_boxes->getOutput(0)}, strides);
    ITensor* concat_bs[] = {yolo->getOutput(0), trans_scores->getOutput(0)};
    IConcatenationLayer* result = network->addConcatenation(concat_bs, 2);
    result->setAxis(2);
    result->setName("Output Tensor");
    ITensor* final_out_tensor = result->getOutput(0);

    // output
    final_out_tensor->setName(kOutputTensorName);
    network->markOutput(*final_out_tensor);

    // verbose
    print_network_layers(network);
    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // delete heap
    delete network;

    return serialized_model;
}

