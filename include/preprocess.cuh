#pragma once
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "config.h"

extern "C" void cuda_preprocess(cv::Mat& img, float*& dst, cudaStream_t& stream);

void cuda_preprocess_init();

void cuda_preprocess_destroy();
