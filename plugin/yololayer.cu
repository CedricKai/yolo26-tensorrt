#include <assert.h>
#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include <cuda_fp16.h>
#include "cuda_utils.h"
#include "yololayer.h"

__global__ void CalDetectionF32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_point,
    const float* __restrict__ anchor_cx_dev,
    const float* __restrict__ anchor_cy_dev,
    const float* __restrict__ stride_dev,
    int64_t num_point_per_batch
)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_point) return;

    int64_t p = idx % num_point_per_batch;
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4 in_val = in4[idx];

    float s = stride_dev[p];
    float cx = anchor_cx_dev[p];
    float cy = anchor_cy_dev[p];

    float4 out_val;
    out_val.x = (cx - in_val.x) * s;
    out_val.y = (cy - in_val.y) * s;
    out_val.z = (cx + in_val.z) * s;
    out_val.w = (cy + in_val.w) * s;

    float4* out4 = reinterpret_cast<float4*>(output);
    out4[idx] = out_val;
}

namespace nvinfer1 {

YoloLayerPlugin::YoloLayerPlugin(
    int classCount,
    int netWidth, int netHeight, int maxOut,
    const int* strides, int stridesLength)
    : mClassCount(classCount)
    , mInputWidth(netWidth)
    , mInputHeight(netHeight)
    , mMaxOutObject(maxOut)
    , mStrides(strides, strides + stridesLength)
{
    initFieldsToSerialize();
    host2dev();
}

YoloLayerPlugin::~YoloLayerPlugin() {
    if(mAnchorCxDev) { cudaFree(mAnchorCxDev); mAnchorCxDev = nullptr; }
    if(mAnchorCyDev) { cudaFree(mAnchorCyDev); mAnchorCyDev = nullptr; }
    if(mStrideDev)   { cudaFree(mStrideDev);   mStrideDev = nullptr; }
}

void YoloLayerPlugin::initFieldsToSerialize()
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("classCount", &mClassCount, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("netWidth", &mInputWidth, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("netHeight", &mInputHeight, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("maxOut", &mMaxOutObject, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("strides", mStrides.data(), PluginFieldType::kINT32, static_cast<int32_t>(mStrides.size()));

    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
}

void YoloLayerPlugin::host2dev() {
    int nPerBatch = 0;
    for(auto& val : mStrides)
    {
        int gw = mInputWidth / val;
        int gh = mInputHeight / val;
        nPerBatch += gw * gh;
    }
    mAnchorCxHost.reserve(nPerBatch);
    mAnchorCyHost.reserve(nPerBatch);
    mStrideHost.reserve(nPerBatch);

    const float grid_cell_offset = 0.5f;
    for (auto& stride : mStrides)
    {
        int H = mInputHeight / stride;
        int W = mInputWidth / stride;

        for (int y = 0; y < H; y++)
        {
            float cy = y + grid_cell_offset;
            for (int x = 0; x < W; x++)
            {
                float cx = x + grid_cell_offset;
                mAnchorCxHost.push_back(cx);
                mAnchorCyHost.push_back(cy);
                mStrideHost.push_back(static_cast<float>(stride));
            }
        }
    }

    assert(static_cast<int>(mAnchorCxHost.size()) == nPerBatch);

    size_t elemCount = mAnchorCxHost.size();
    size_t bytes = elemCount * sizeof(float);

    CUDA_CHECK(cudaMalloc(&mAnchorCxDev, bytes));
    CUDA_CHECK(cudaMalloc(&mAnchorCyDev, bytes));
    CUDA_CHECK(cudaMalloc(&mStrideDev, bytes));

    CUDA_CHECK(cudaMemcpy(mAnchorCxDev, mAnchorCxHost.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mAnchorCyDev, mAnchorCyHost.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mStrideDev, mStrideHost.data(), bytes, cudaMemcpyHostToDevice));
}

IPluginCapability* YoloLayerPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
        return static_cast<IPluginV3OneBuild*>(this);
    if (type == PluginCapabilityType::kRUNTIME)
        return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* YoloLayerPlugin::clone() noexcept
{
    auto obj = std::make_unique<YoloLayerPlugin>(*this);

    obj->mAnchorCxDev = nullptr;
    obj->mAnchorCyDev = nullptr;
    obj->mStrideDev = nullptr;

    size_t elemCount = obj->mAnchorCxHost.size();
    size_t bytes = elemCount * sizeof(float);

    cudaError_t err0 = cudaMalloc(&obj->mAnchorCxDev, bytes);
    cudaError_t err1 = cudaMalloc(&obj->mAnchorCyDev, bytes);
    cudaError_t err2 = cudaMalloc(&obj->mStrideDev, bytes);
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess)
    {
        // Clean up any partially allocated memory to prevent leaks
        if (obj->mAnchorCxDev) { cudaFree(obj->mAnchorCxDev); }
        if (obj->mAnchorCyDev) { cudaFree(obj->mAnchorCyDev); }
        if (obj->mStrideDev)   { cudaFree(obj->mStrideDev); }
        return nullptr;
    }

    CUDA_CHECK(cudaMemcpy(obj->mAnchorCxDev, obj->mAnchorCxHost.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(obj->mAnchorCyDev, obj->mAnchorCyHost.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(obj->mStrideDev, obj->mStrideHost.data(), bytes, cudaMemcpyHostToDevice));

    return obj.release();
}

char const* YoloLayerPlugin::getPluginName() const noexcept
{ return "YoloLayer_TRT"; }

char const* YoloLayerPlugin::getPluginVersion() const noexcept
{ return "1"; }

char const* YoloLayerPlugin::getPluginNamespace() const noexcept
{ return ""; }

int32_t YoloLayerPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t YoloLayerPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

bool YoloLayerPlugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    DataType dtype = inOut[pos].desc.type;
    PluginFormat pfmt = inOut[pos].desc.format;

    bool ret = false;
    if (pos < nbInputs)
    {
        bool typeOk = (dtype == DataType::kFLOAT);
        bool fmtOk  = (pfmt == PluginFormat::kLINEAR);
        ret = typeOk && fmtOk;
    }
    else
    {
        bool typeOk = (dtype == DataType::kFLOAT);
        bool fmtOk  = (pfmt == PluginFormat::kLINEAR);
        ret = typeOk && fmtOk;
    }
    return ret;
}

int32_t YoloLayerPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if(nbOutputs != 1)
        return -1;
    outputTypes[0] = DataType::kFLOAT;
    return 0;
}

int32_t YoloLayerPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs,
    DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    int totalPoints = 0;
    for(auto s : mStrides)
    {
        int gw = mInputWidth / s;
        int gh = mInputHeight / s;
        totalPoints += gw * gh;
    }
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(totalPoints);
    outputs[0].d[2] = exprBuilder.constant(4);
    return 0;
}

int32_t YoloLayerPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    int64_t batchSize = inputDesc[0].dims.d[0];
    const auto& inDims = inputDesc[0].dims;

    int64_t totalPoint = 1;
    for(int i = 0; i < inDims.nbDims; i++)
    {
        totalPoint *= inDims.d[i];
    }
    int64_t pointPerBatch = totalPoint / batchSize;

    forwardGpu(inputs, reinterpret_cast<float*>(outputs[0]), stream, batchSize, totalPoint, pointPerBatch);
    return 0;
}

int32_t YoloLayerPlugin::onShapeChange(PluginTensorDesc const* in, int32_t nbInputs,
    PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

IPluginV3* YoloLayerPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* YoloLayerPlugin::getFieldsToSerialize() noexcept
{
    return &mFCToSerialize;
}

size_t YoloLayerPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void YoloLayerPlugin::forwardGpu(const void* const* inputs, float* output,
    cudaStream_t stream, int batchSize, int64_t totalPoint, int64_t pointPerBatch)
{
    dim3 block(mThreadCount,1U,1U);
    dim3 grid(((totalPoint + block.x - 1)/ block.x), 1U,1U);

    CalDetectionF32<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(inputs[0]),
        output,
        totalPoint,
        mAnchorCxDev,
        mAnchorCyDev,
        mStrideDev,
        pointPerBatch
    );
}

YoloPluginCreator::YoloPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("combinedInfo", nullptr, PluginFieldType::kINT32, 0);
    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* YoloPluginCreator::getPluginName() const noexcept
{ return "YoloLayer_TRT"; }

char const* YoloPluginCreator::getPluginVersion() const noexcept
{ return "1"; }

char const* YoloPluginCreator::getPluginNamespace() const noexcept
{ return ""; }

PluginFieldCollection const* YoloPluginCreator::getFieldNames() noexcept
{ return &mFC; }

IPluginV3* YoloPluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        if (phase == TensorRTPhase::kBUILD)
        {
            int class_count = 0;
            int input_w = 0;
            int input_h = 0;
            int max_out = 0;
            std::vector<int> strides;

            for (int i = 0; i < fc->nbFields; i++)
            {
                const auto& field = fc->fields[i];
                if (std::strcmp(field.name, "classCount") == 0)
                {
                    assert(field.type == nvinfer1::PluginFieldType::kINT32);
                    assert(field.length == 1);
                    class_count = *(static_cast<const int*>(field.data));
                }
                else if (std::strcmp(field.name, "netWidth") == 0)
                {
                    assert(field.type == nvinfer1::PluginFieldType::kINT32);
                    assert(field.length == 1);
                    input_w = *(static_cast<const int*>(field.data));
                }
                else if (std::strcmp(field.name, "netHeight") == 0)
                {
                    assert(field.type == nvinfer1::PluginFieldType::kINT32);
                    assert(field.length == 1);
                    input_h = *(static_cast<const int*>(field.data));
                }
                else if (std::strcmp(field.name, "maxOut") == 0)
                {
                    assert(field.type == nvinfer1::PluginFieldType::kINT32);
                    assert(field.length == 1);
                    max_out = *(static_cast<const int*>(field.data));
                }
                else if (std::strcmp(field.name, "strides") == 0)
                {
                    assert(field.type == nvinfer1::PluginFieldType::kINT32);
                    const int* p_stride = static_cast<const int*>(field.data);
                    strides.assign(p_stride, p_stride + field.length);
                }
            }
            auto* obj = new YoloLayerPlugin(class_count, input_w, input_h, max_out, strides.data(), static_cast<int>(strides.size()));
            return obj;
        }
        else if (phase == TensorRTPhase::kRUNTIME)
        {
            int classCount = 0;
            int netWidth = 0;
            int netHeight = 0;
            int maxOut = 0;
            std::vector<int> strides;

            for (int i = 0; i < fc->nbFields; i++) {
                auto const& f = fc->fields[i];
                if(std::strcmp(f.name, "classCount") == 0)
                    classCount = *static_cast<const int*>(f.data);
                else if(std::strcmp(f.name, "netWidth") == 0)
                    netWidth = *static_cast<const int*>(f.data);
                else if(std::strcmp(f.name, "netHeight") == 0)
                    netHeight = *static_cast<const int*>(f.data);
                else if(std::strcmp(f.name, "maxOut") == 0)
                    maxOut = *static_cast<const int*>(f.data);
                else if(std::strcmp(f.name, "strides") == 0)
                {
                    auto p = static_cast<const int*>(f.data);
                    strides.assign(p, p + f.length);
                }
            }
            auto* obj = new YoloLayerPlugin(classCount, netWidth, netHeight, maxOut, strides.data(), static_cast<int>(strides.size()));
            return obj;
        }
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
    return nullptr;
}

}
