#pragma once
#include <string>
#include <vector>
#include "macros.h"

namespace nvinfer1 {

/**
 * @brief YOLO26 postprocess plugin for TensorRT10.x IPluginV3
 * Decode bbox offset from feature grid, output decoded box coordinates
 */
class API YoloLayerPlugin :
    public IPluginV3,
    public IPluginV3OneCore,
    public IPluginV3OneBuild,
    public IPluginV3OneRuntime
{
public:
    /**
     * @brief Copy constructor, used for clone and attachToContext
     * @param p source plugin instance
     */
    YoloLayerPlugin(YoloLayerPlugin const& p) = default;

    /**
     * @brief Main constructor invoked when building network
     * @param classCount number of object classes
     * @param netWidth network input image width
     * @param netHeight network input image height
     * @param maxOut maximum output bounding boxes per image
     * @param strides pointer of stride values for each detection head
     * @param stridesLength number of stride elements
     */
    YoloLayerPlugin(
        int classCount,
        int netWidth,
        int netHeight,
        int maxOut,
        const int* strides,
        int stridesLength);

    ~YoloLayerPlugin() override;

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
        DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
        DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs,
        PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

private:
    void initFieldsToSerialize();
    void host2dev();

    void forwardGpu(const void* const* inputs, float* output, cudaStream_t stream,
        int batchSize, int64_t totalPoint, int64_t pointPerBatch);

    int mClassCount;
    int mInputWidth;
    int mInputHeight;
    int mMaxOutObject;
    std::vector<int> mStrides;

    int mThreadCount{256};
    float mObjThreshold{0.4f};

    std::vector<float> mAnchorCxHost;
    std::vector<float> mAnchorCyHost;
    std::vector<float> mStrideHost;
    float* mAnchorCxDev{nullptr};
    float* mAnchorCyDev{nullptr};
    float* mStrideDev{nullptr};

    std::vector<PluginField> mDataToSerialize;
    PluginFieldCollection mFCToSerialize;
};

/**
 * @brief Plugin creator factory for YoloLayer_TRT, IPluginCreatorV3One for TensorRT10
 */
class API YoloPluginCreator : public IPluginCreatorV3One {
public:
    YoloPluginCreator();
    ~YoloPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc,
        TensorRTPhase phase) noexcept override;

private:
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFC;
};

REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
}
