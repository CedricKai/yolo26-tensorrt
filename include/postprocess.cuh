#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include "cuda_utils.h"
#include "config.h"

extern "C" void cuda_postprocess(float*& d_input, float*& d_output, int img0_w, int img0_h, cudaStream_t& stream);

void cuda_postprocess_init();

void cuda_postprocess_destroy();
