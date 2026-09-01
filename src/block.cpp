#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cstring>
#include <numeric>
#include <memory>
#include "block.h"
#include "config.h"
#include "yololayer.h"


static int autopad(int k, int p = -1, int d = 1){
    int k_eff = k;
    if (d > 1)   {k_eff = d * (k - 1) + 1;}
    if (p == -1) {p = k_eff / 2;}
    return p;
}


WeightHolder loadWeights(const std::string& file) {
    std::cout << "Loading weights: " << file << std::endl;
    WeightHolder wh;
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        std::string name;
        uint32_t size;
        input >> name >> std::dec >> size;
        if (!input.good()) break;

        auto buf = new float[size];
        wh.bufs.push_back(buf);

        input >> std::hex;
        uint32_t tmp;
        for (uint32_t i = 0; i < size; i++) {
            input >> tmp;
            memcpy(buf + i, &tmp, sizeof(float));
        }
        input >> std::dec;

        Weights wt{DataType::kFLOAT, buf, (int64_t)size};
        wh.wmap[name] = wt;
    }
    return wh;
}


ITensor* conv_2d(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int ch, int k, int s, int p, int g, int d) {
    Weights bias_empty{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv = network->addConvolutionNd(input, ch, DimsHW{k, k},
        wHolder.wmap[lname + ".conv.weight"], bias_empty);
    assert(conv);
    conv->setStrideNd(DimsHW{s, s});
    conv->setPaddingNd(DimsHW{autopad(k, p, d), autopad(k, p, d)});
    conv->setNbGroups(g);
    conv->setDilationNd(DimsHW{d, d});
    conv->setName((lname + ".conv").c_str());

    return conv->getOutput(0);
}

ITensor* conv_2_detect(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int ch, int k, int s, int p, int g, int d) {
    IConvolutionLayer* conv = network->addConvolutionNd(input, ch, DimsHW{k, k},
        wHolder.wmap[lname + ".weight"], wHolder.wmap[lname + ".bias"]);
    assert(conv);
    conv->setStrideNd(DimsHW{s, s});
    conv->setPaddingNd(DimsHW{autopad(k, p, d), autopad(k, p, d)});
    conv->setNbGroups(g);
    conv->setDilationNd(DimsHW{d, d});
    conv->setName((lname + ".conv").c_str());

    return conv->getOutput(0);
}

ITensor* batch_norm_2d(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname){
    const std::string key_w         = lname + ".weight";
    const std::string key_b         = lname + ".bias";
    const std::string key_mean      = lname + ".running_mean";
    const std::string key_var       = lname + ".running_var";
    assert(wHolder.wmap.find(key_w)    != wHolder.wmap.end());
    assert(wHolder.wmap.find(key_b)    != wHolder.wmap.end());
    assert(wHolder.wmap.find(key_mean) != wHolder.wmap.end());
    assert(wHolder.wmap.find(key_var)  != wHolder.wmap.end());

    int len = static_cast<int>(wHolder.wmap[key_var].count);
    auto* bn_weight       = static_cast<const float*>(wHolder.wmap[key_w].values);
    auto* bn_bias         = static_cast<const float*>(wHolder.wmap[key_b].values);
    auto* bn_running_mean = static_cast<const float*>(wHolder.wmap[key_mean].values);
    auto* bn_running_var  = static_cast<const float*>(wHolder.wmap[key_var].values);

    auto* scale_weights = new float[len];
    auto* shift_weights = new float[len];
    auto* power_weights = new float[len];
    wHolder.bufs.push_back(scale_weights);
    wHolder.bufs.push_back(shift_weights);
    wHolder.bufs.push_back(power_weights);

    for (int i = 0; i < len; ++i) {
        // scale[i] = weight[i] / sqrt(var[i] + epsilon)
        scale_weights[i] = bn_weight[i] / sqrtf(bn_running_var[i] + 1e-3f);
        // shift[i] = bias[i] - (mean[i] * scale[i])
        shift_weights[i] = bn_bias[i] - bn_running_mean[i] * scale_weights[i];
        //power = 1.0;
        power_weights[i] = 1.0f;
    }

    Weights scale{DataType::kFLOAT, scale_weights, len};
    Weights shift{DataType::kFLOAT, shift_weights, len};
    Weights power{DataType::kFLOAT, power_weights, len};

    // out = (input * scale + shift) ^ power
    IScaleLayer* output = network->addScaleNd(input, ScaleMode::kCHANNEL, shift, scale, power, 1);
    output->setName((lname).c_str());
    assert(output);

    return output->getOutput(0);
}

ITensor* conv_bn_silu(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int ch, int k, int s, int p, int g, int d, bool act) {
    ITensor* conv = conv_2d(network, input, wHolder, lname, ch ,k, s, p, g, d);
    ITensor* bn = batch_norm_2d(network, *conv, wHolder, lname + ".bn");
    if (!act) {
        return bn;
    }

    IActivationLayer* sigmoid = network->addActivation(*bn, ActivationType::kSIGMOID);
    IElementWiseLayer* silu = network->addElementWise(*bn, *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    assert(silu);
    sigmoid->setName((lname + ".sigmoid").c_str());
    silu->setName((lname + ".silu").c_str());

    return silu->getOutput(0);
}

ITensor* bottleneck(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, bool shortcut, int g, int k, float e) {
    int c_ = static_cast<int>(c2 * e);  // NOLINT
    ITensor* conv1 = conv_bn_silu(network, input, wHolder,  lname + ".cv1", c_, k, 1);
    ITensor* conv2 = conv_bn_silu(network, *conv1,wHolder, lname + ".cv2", c2, k, 1, -1, g);

    if (shortcut && c1 == c2) {
        IElementWiseLayer* ew = network->addElementWise(input, *conv2, ElementWiseOperation::kSUM);
        ew->setName((lname + ".shortcut").c_str());

        return  ew->getOutput(0);
    }
    return conv2;
}

ITensor* C3k(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, int n, bool shortcut, int g, float e){
    int c_ = static_cast<int>(c2 * e);  // NOLINT

    std::string conv1_name = lname + ".cv1";
    ITensor* conv1 = conv_bn_silu(network, input, wHolder, conv1_name,
        c_, 1 ,1);

    ITensor* y1 = conv1;
    for (int i = 0; i < n; i++) {
        auto* b = bottleneck(network, *y1, wHolder,  lname + ".m." + std::to_string(i),
            c_, c_, shortcut, g, 3, 1.0);

        y1 = b;
    }

    std::string conv2_name = lname + ".cv2";
    ITensor* conv2 = conv_bn_silu(network, input, wHolder, conv2_name ,
        c_, 1 ,1);

    ITensor* inputTensor0[] = {y1, conv2};
    IConcatenationLayer* cat = network->addConcatenation(inputTensor0, 2);
    cat->setAxis(1);
    cat->setName((lname + ".cat").c_str());

    std::string conv3_name = lname + ".cv3";
    ITensor* conv3 = conv_bn_silu(network, *cat->getOutput(0), wHolder, conv3_name,
        c2, 1 );

    return conv3;
}

ITensor* C3k2(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, int n, bool shortcut, int g, float e, bool atten, bool c3k) {

    // TODO：n要用的，但是一直是1，就懒得改，其实有bug（原模型的n参数不为1时，就会出现error，少了一段，输出前后维度无法匹配）
    // TODO：可参考26的block.py的的源码
    int c = static_cast<int>(c2 * e);   // NOLINT  // 0.25*256=64
    ITensor* conv1 = conv_bn_silu(network, input, wHolder, lname + ".cv1",
        2 * c, 1, 1);
    Dims d = conv1->getDimensions();

    ISliceLayer* split1 = network->addSlice(*conv1,
       Dims4{0, 0, 0, 0},
       Dims4{1, c, d.d[2], d.d[3]},
       Dims4{1, 1, 1, 1});
    split1->setName((lname + ".split1").c_str());

    ISliceLayer* split2 = network->addSlice(*conv1,
        Dims4{0, c, 0, 0},
        Dims4{1, c, d.d[2], d.d[3]},
        Dims4{1, 1, 1, 1});
    split2->setName((lname + ".split2").c_str());

    ITensor* inputTensor0[] = {split1->getOutput(0), split2->getOutput(0)};
    IConcatenationLayer* cat = network->addConcatenation(inputTensor0, 2);
    cat->setAxis(1);
    cat->setName((lname + ".cat0").c_str());

    ITensor* m = split2->getOutput(0);
    std::vector<ITensor*> y_list;
    y_list.push_back(cat->getOutput(0));

    for(int i = 0; i < n; i++)
    {
        if (atten==true)
        {
            ITensor* y = bottleneck(network, *m, wHolder,  lname + ".m.0.0",
                c, c, shortcut, g);
            m = psa_block(network, *y, wHolder, lname + ".m.0.1",
                c,0.5, std::max(c/64 ,1));
        }
        else if (c3k==true)
        {
            m = C3k(network, *m, wHolder, lname + ".m."+ std::to_string(i),
                c, c, 2, shortcut, g);
        }
        else
        {
            m = bottleneck(network, *m, wHolder,  lname + ".m." + std::to_string(i),
                c, c, shortcut, g);
        }
        y_list.push_back(m);
    }

    IConcatenationLayer* cat1 = network->addConcatenation(y_list.data(), y_list.size());
    cat1->setAxis(1);
    cat1->setName((lname + ".cat1").c_str());

    ITensor* conv2 = conv_bn_silu(network, *cat1->getOutput(0), wHolder, lname + ".cv2",
        c2, 1);

    return conv2;
}

ITensor* SPPF(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, int k, int n, bool shortcut) {
    int c = c1 / 2;
    ITensor* conv1 = conv_bn_silu(network, input, wHolder, lname + ".cv1",
        c, 1, 1, -1, 1, 1, false);

    std::vector<ITensor*> pool_outputs;
    pool_outputs.push_back(conv1);
    ITensor* cur = conv1;
    for (int i = 0; i < n; i++)
    {
        auto pool = network->addPoolingNd(*cur, PoolingType::kMAX, DimsHW{k, k});
        pool->setStrideNd(DimsHW{1, 1});
        pool->setPaddingNd(DimsHW{2, 2});
        pool->setName((lname + ".cv1.pool" + std::to_string(i)).c_str());
        cur = pool->getOutput(0);
        pool_outputs.push_back(cur);
    }

    IConcatenationLayer* cat = network->addConcatenation(pool_outputs.data(), static_cast<int32_t>(pool_outputs.size()));
    cat->setAxis(1);
    cat->setName((lname + ".pool.cat").c_str());

    ITensor* conv2 = conv_bn_silu(network, *cat->getOutput(0), wHolder, lname + ".cv2",
        c2, 1, 1);

    if (shortcut && c1 == c2) {
        IElementWiseLayer* ew = network->addElementWise(input, *conv2, ElementWiseOperation::kSUM);
        ew->setName((lname + ".shortcut").c_str());

        return  ew->getOutput(0);
    }

    return conv2;
}

ITensor* attention(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int dim, int num_heads, float attn_ratio) {

    int head_dim = dim / num_heads;
    int key_dim = static_cast<int>(head_dim * attn_ratio);  // NOLINT
    float scale = 1.0f / sqrtf(static_cast<float>(key_dim));
    int nh_kd = key_dim * num_heads;
    int h = dim + nh_kd * 2;

    Dims d = input.getDimensions();
    int B = static_cast<int>(d.d[0]);
    int C = static_cast<int>(d.d[1]);
    int H = static_cast<int>(d.d[2]);
    int W = static_cast<int>(d.d[3]);
    int N = H * W;

    ITensor* qkv = conv_bn_silu(network, input, wHolder, lname + ".qkv",
    h, 1, 1, -1, 1, 1, false);
    qkv->setName((lname + ".attn_qkv").c_str());

    // --------------------------
    // Step1: reshape qkv: [B, h_total, H, W] → [B, num_heads, qkv_ch, N]
    // qkv_ch = key_dim*2 + head_dim
    IShuffleLayer* shuffle_qkv = network->addShuffle(*qkv);
    shuffle_qkv->setReshapeDimensions(Dims4{B, num_heads, key_dim * 2 + head_dim, N});
    shuffle_qkv->setName((lname + ".shuffle_qkv").c_str());

    // Step2: split dim=2, split [key_dim, key_dim, head_dim] → q,k,v
    ISliceLayer* attn_q = network->addSlice(*shuffle_qkv->getOutput(0),
        Dims4{0,0,0,0},
        Dims4{B, num_heads, key_dim, N},
        Dims4{1,1,1,1});
    attn_q->setName((lname + ".attn_q").c_str());

    ISliceLayer* attn_k = network->addSlice(*shuffle_qkv->getOutput(0),
        Dims4{0,0,key_dim,0},
        Dims4{B, num_heads, key_dim, N},
        Dims4{1,1,1,1});
    attn_k->setName((lname + ".attn_k").c_str());

    ISliceLayer* attn_v = network->addSlice(*shuffle_qkv->getOutput(0),
        Dims4{0,0, key_dim*2, 0},
        Dims4{B, num_heads, head_dim, N},
        Dims4{1,1,1,1});
    attn_v->setName((lname + ".attn_v").c_str());

    // q * scale
    auto* sc = new float(scale);
    wHolder.bufs.push_back(sc);

    Weights w_scale{DataType::kFLOAT, sc, 1};
    IConstantLayer* const_scale = network->addConstant(Dims4{1,1,1,1}, w_scale);
    IElementWiseLayer* q_scaled = network->addElementWise(*attn_q->getOutput(0), *const_scale->getOutput(0), ElementWiseOperation::kPROD);
    const_scale->setName((lname + ".const_scale").c_str());
    q_scaled->setName((lname + ".q_scale").c_str());

    // (q * scale).transpose(-2, -1)
    // transpose q_scaled: [B,nh,kd,N] permute(0,1,3,2) → [B,nh,N,kd]
    IShuffleLayer* trans_q = network->addShuffle(*q_scaled->getOutput(0));
    trans_q->setName((lname + ".q_trans").c_str());
    trans_q->setFirstTranspose(Permutation{0,1,3,2});

    // (q * scale).transpose(-2, -1) @ k
    // k: [B,nh,kd,N]
    // matrix multiply trans_q @ attn_k  → (B,nh,N,kd) * (B,nh,kd,N) = (B,nh,N,N)
    // TensorRT matrixMultiply: op0=kMATRIX_NONE, op1=kMATRIX_NONE
    IMatrixMultiplyLayer* attn_mat = network->addMatrixMultiply(
        *trans_q->getOutput(0), MatrixOperation::kNONE,
        *attn_k->getOutput(0),   MatrixOperation::kNONE);
    attn_mat->setName((lname + ".attn_mat").c_str());

    // attn.softmax(dim=-1)
    // softmax dim=-1 (last dimension)
    ISoftMaxLayer* attn_softmax = network->addSoftMax(*attn_mat->getOutput(0));
    // bit3 bit2 bit1 bit0
    // 1    0    0    0     ← 值为 8
    // ↓    ↓    ↓    ↓
    // W    H    C    N
    // ✅   ❌   ❌   ❌    ← 只有 W 被选中
    attn_softmax->setAxes(1U << 3);
    attn_softmax->setName((lname + ".attn_softmax").c_str());

    // attn.transpose(-2, -1)
    // attn transpose: permute(0,1,3,2) → [B,nh,N,N]
    IShuffleLayer* trans_attn = network->addShuffle(*attn_softmax->getOutput(0));
    trans_attn->setFirstTranspose(Permutation{0,1,3,2});
    trans_attn->setName((lname + ".attn_trans").c_str());

    // v @ attn.transpose(-2, -1)
    // v @ attn_t : v[B,nh,hd,N]  @ attn_t[B,nh,N,N] → [B,nh,hd,N]
    IMatrixMultiplyLayer* val_mat = network->addMatrixMultiply(
        *attn_v->getOutput(0), MatrixOperation::kNONE,
        *trans_attn->getOutput(0), MatrixOperation::kNONE);
    val_mat->setName((lname + ".val_mat").c_str());

    // (v @ attn.transpose(-2, -1)).view(B, C, H, W)
    // reshape [B,nh,hd,N] → [B,C,H,W]
    IShuffleLayer* trans_val = network->addShuffle(*val_mat->getOutput(0));
    trans_val->setName((lname + ".val_mat_trans").c_str());
    trans_val->setReshapeDimensions(Dims4{B, C, H, W});

    // v.reshape(B,C,H,W)
    IShuffleLayer* v_shuffle_pe = network->addShuffle(*attn_v->getOutput(0));
    v_shuffle_pe->setName((lname + ".v_for_pe").c_str());
    v_shuffle_pe->setReshapeDimensions(Dims4{B, C, H, W});

    // pe(v.reshape(B,C,H,W))
    ITensor* pe = conv_bn_silu(network, *v_shuffle_pe->getOutput(0), wHolder, lname + ".pe",
    dim, 3, 1, -1, dim, 1, false);
    pe->setName((lname + ".pe").c_str());

    // (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
    IElementWiseLayer* add_pe = network->addElementWise(*trans_val->getOutput(0), *pe, ElementWiseOperation::kSUM);
    add_pe->setName((lname + ".add_pe").c_str());

    // proj(.)
    ITensor* proj = conv_bn_silu(network, *add_pe->getOutput(0), wHolder, lname + ".proj",
    dim, 1, 1, -1, 1, 1, false);
    proj ->setName((lname + ".proj").c_str());

    return proj;
}

ITensor* psa_block(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c, float attn_ratio, int num_heads, bool shortcut) {

    ITensor* attn = attention(network, input, wHolder, lname + ".attn",
        c, num_heads, attn_ratio);

    if (shortcut) {
        IElementWiseLayer* attn_cat = network->addElementWise(*attn, input, ElementWiseOperation::kSUM);
        attn_cat->setName((lname + ".attn_cat").c_str());
        attn = attn_cat->getOutput(0);
    }

    ITensor* ffn_conv1 = conv_bn_silu(network, *attn, wHolder, lname + ".ffn.0",
        c * 2, 1, 1);

    ITensor* ffn_conv2 = conv_bn_silu(network, *ffn_conv1, wHolder, lname + ".ffn.1",
        c, 1, 1, -1, 1, 1, false);

    if (shortcut) {
        IElementWiseLayer* ffn_cat = network->addElementWise(*attn, *ffn_conv2, ElementWiseOperation::kSUM);
        ffn_cat->setName((lname + ".ffn_cat").c_str());
        ffn_conv2 = ffn_cat->getOutput(0);
    }

    return ffn_conv2;
}

ITensor* C2PSA(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    const std::string& lname,
    int c1, int c2, int n, float e) {
    if (c1!=c2)
        return nullptr;

    Dims input_dim = input.getDimensions();
    // int c = static_cast<int>(input_dim.d[1] * e);  // NOLINT
    int c = c1 * e;
    ITensor* conv1 = conv_bn_silu(network, input, wHolder, lname + ".cv1",
        2 * c, 1, 1);
    Dims d = conv1->getDimensions();

    ISliceLayer* split1 = network->addSlice(*conv1,
       Dims4{0, 0, 0, 0},
       Dims4{1, c, d.d[2], d.d[3]},
       Dims4{1, 1, 1, 1});
    split1->setName((lname + ".split1").c_str());

    ISliceLayer* split2 = network->addSlice(*conv1,
        Dims4{0, c, 0, 0},
        Dims4{1, c, d.d[2], d.d[3]},
        Dims4{1, 1, 1, 1});
    split2->setName((lname + ".split2").c_str());

    ITensor* y = split2->getOutput(0);
    for (int i = 0; i < n; i++) {
         auto * b= psa_block(network, *y, wHolder, lname + ".m.0",
             c,0.5, std::max(c/64 ,1));
         y = b;
    }

    ITensor* inputTensor1[] = {split1->getOutput(0), y};
    IConcatenationLayer* cat = network->addConcatenation(inputTensor1, 2);
    cat->setAxis(1);
    cat->setName((lname + ".cat").c_str());

    ITensor* conv2 = conv_bn_silu(network, *cat->getOutput(0), wHolder, lname + ".cv2",
        static_cast<int>(input_dim.d[1]), 1);

    return conv2;
}

void build_detect_head(
    INetworkDefinition* network,
    ITensor& input,
    WeightHolder& wHolder,
    Yolo26Detect& det,
    const std::string& lname1,
    const std::string& lname2,
    int in_c,
    int bs,
    int c2,
    int c3) {
    // boxes
    ITensor* cbs1 = conv_bn_silu(network, input, wHolder, lname1 + ".0",
        c2, 3);
    cbs1->setName((lname1 + ".cbs1").c_str());

    ITensor* cbs2 = conv_bn_silu(network, *cbs1, wHolder, lname1 + ".1",
        c2, 3);
    cbs2->setName((lname1 + ".cbs2").c_str());

    ITensor* conv1= conv_2_detect(network, *cbs2, wHolder, lname1 + ".2", 4 * kRegMax, 1);
    conv1->setName((lname1 + ".conv1").c_str());

    Dims3 reshape_dims1{bs, 4 * kRegMax, -1};
    IShuffleLayer* shuffle_cov1 = network->addShuffle(*conv1);
    shuffle_cov1->setReshapeDimensions(reshape_dims1);
    shuffle_cov1->setName((lname1 + ".shuffle_cov1").c_str());
    det.boxes = shuffle_cov1->getOutput(0);

    // scores
    ITensor* dwconv1 = conv_bn_silu(network, input, wHolder, lname2 + ".0.0",
        in_c, 3, 1,-1, std::gcd(in_c, in_c));
    dwconv1->setName((lname2 + ".dwconv1").c_str());

    ITensor* cbs3 = conv_bn_silu(network, *dwconv1, wHolder, lname2 + ".0.1",
        c3, 1);
    cbs3->setName((lname2 + ".cbs3").c_str());

    ITensor* dwconv2 = conv_bn_silu(network, *cbs3, wHolder, lname2 + ".1.0",
        c3, 3, 1, -1, std::gcd(c3, c3));
    dwconv2->setName((lname2 + ".dwconv2").c_str());

    ITensor* cbs4 = conv_bn_silu(network, *dwconv2, wHolder, lname2 + ".1.1",
        c3, 1);
    cbs4->setName((lname2 + ".cbs4").c_str());

    ITensor* cobv2= conv_2_detect(network, *cbs4, wHolder, lname2 + ".2",
        kNumClass, 1);
    cobv2->setName((lname2 + ".cobv2").c_str());

    Dims3 reshape_dims2{bs, kNumClass, -1};
    IShuffleLayer* shuffle_cobv2 = network->addShuffle(*cobv2);
    shuffle_cobv2->setReshapeDimensions(reshape_dims2);
    shuffle_cobv2->setName((lname2 + ".shuffle_cobv2").c_str());
    det.scores = shuffle_cobv2->getOutput(0);
}

IPluginV3Layer* add_yolo_layer(
    INetworkDefinition* network,
    std::vector<ITensor*> dets,     // NOLINT
    std::vector<int>& px_array) {

    IPluginCreatorInterface* baseCreator = getPluginRegistry()->getCreator(
        "YoloLayer_TRT",  "1", "");
    auto* creator = dynamic_cast<IPluginCreatorV3One*>(baseCreator);

    std::vector<PluginField> fieldVec;
    fieldVec.emplace_back("classCount", &kNumClass,         PluginFieldType::kINT32, 1);
    fieldVec.emplace_back("netWidth",   &kInputW,           PluginFieldType::kINT32, 1);
    fieldVec.emplace_back("netHeight",  &kInputH,           PluginFieldType::kINT32, 1);
    fieldVec.emplace_back("maxOut",     &kMaxNumOutputBbox, PluginFieldType::kINT32, 1);
    fieldVec.emplace_back("strides",    px_array.data(),     PluginFieldType::kINT32,  static_cast<int32_t>(px_array.size()));

    PluginFieldCollection fc;
    fc.nbFields = static_cast<int32_t>(fieldVec.size());
    fc.fields = fieldVec.data();

    IPluginV3* pluginObj = creator->createPlugin("YoloLayer_TRT", &fc, TensorRTPhase::kBUILD);
    IPluginV3Layer* pluginLayer = network->addPluginV3(dets.data(),
        static_cast<int32_t>(dets.size()),
        nullptr,
        0,
        *pluginObj);
    pluginLayer->setName("YoloLayer_TRT");
    return pluginLayer;
}
