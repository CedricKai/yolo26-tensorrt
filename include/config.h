#pragma once

enum class PrecisionMode
{
    FP32,
    FP16,
    INT8
};

struct alignas(float) Detection {
    float left;
    float top;
    float right;
    float bottom;
    float conf;
    float class_id;
};

inline constexpr const char* kInputTensorName = "images";
inline constexpr const char* kOutputTensorName = "output";
inline constexpr int kNumClass = 80;
inline constexpr int kInputH = 640;
inline constexpr int kInputW = 640;
inline constexpr int kGpuId = 0;
inline constexpr int kBatchSize = 1;
inline constexpr int kRegMax = 1;
inline constexpr float kConfThresh = 0.45f;
inline constexpr int kMaxNumOutputBbox = 300;
inline constexpr int kMaxNumOutputBboxSIZE = 300 * sizeof(Detection) / sizeof(float);
inline constexpr int kMaxInputImageSize = 1500 * 1500 * 3;
inline constexpr int kWarmUp = 10;
inline constexpr int kVisMax = 30;
inline constexpr const char* kInputQuantizationFolder = "./coco_calib";

inline constexpr int NUM_BOXES = 8400;
inline constexpr int NUM_CLASSES = 80;
inline constexpr int BBOX_DIM = 4;
inline constexpr int INFO_PER_BOX = 84;
inline constexpr int TOP_K = 300;
inline constexpr int OUT_DIM = 6;