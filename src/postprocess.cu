#include "postprocess.cuh"
#include <cub/cub.cuh>

float* d_max_scores = nullptr;
int* d_max_class_ids = nullptr;
int* d_indices = nullptr;
float* d_sorted_scores = nullptr;
int* d_sorted_indices = nullptr;
void* d_sort_temp = nullptr;
size_t d_sort_temp_bytes = 0;


__global__ void argmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ max_scores,
    int* __restrict__ max_class_ids)
{
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= NUM_BOXES) return;

    const float* scores = input + box_idx * INFO_PER_BOX + BBOX_DIM;

    float max_val = -FLT_MAX;
    int max_idx = -1;
    #pragma unroll 4
    for (int c = 0; c < NUM_CLASSES; ++c) {
        float val = scores[c];
        // Sanitize NaN/Inf to prevent CUB sort from producing garbage indices
        if (isnan(val) || isinf(val)) val = -FLT_MAX;
        if (val > max_val) {
            max_val = val;
            max_idx = c;
        }
    }
    max_scores[box_idx] = max_val;
    max_class_ids[box_idx] = max_idx;
}


__global__ void gather_and_format_kernel(
    const float* __restrict__ input,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_scores,
    const int* __restrict__ max_class_ids,
    float* __restrict__ output,
    int img_w, int img_h, float pad_x, float pad_y, float gain) {

    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (k_idx >= TOP_K) return;

    float* out_row = output + k_idx * OUT_DIM;
    float score = topk_scores[k_idx];

    if (score < kConfThresh) {
        out_row[0] = 0.0f;
        out_row[1] = 0.0f;
        out_row[2] = 0.0f;
        out_row[3] = 0.0f;
        out_row[4] = -1.0f;
        out_row[5] = -1.0f;
        return;
    }

    int original_box_idx = topk_indices[k_idx];
    // Bounds check to prevent illegal memory access from garbage indices
    if (original_box_idx < 0 || original_box_idx >= NUM_BOXES) {
        out_row[0] = 0.0f;
        out_row[1] = 0.0f;
        out_row[2] = 0.0f;
        out_row[3] = 0.0f;
        out_row[4] = -1.0f;
        out_row[5] = -1.0f;
        return;
    }

    const float* box_data = input + original_box_idx * INFO_PER_BOX;
    int class_id = max_class_ids[original_box_idx];

    float x1 = fminf(fmaxf((box_data[0] - pad_x) / gain, 0.0f), img_w);
    float y1 = fminf(fmaxf((box_data[1] - pad_y) / gain, 0.0f), img_h);
    float x2 = fminf(fmaxf((box_data[2] - pad_x) / gain, 0.0f), img_w);
    float y2 = fminf(fmaxf((box_data[3] - pad_y) / gain, 0.0f), img_h);

    out_row[0] = x1;
    out_row[1] = y1;
    out_row[2] = x2;
    out_row[3] = y2;
    out_row[4] = score;
    out_row[5] = static_cast<float>(class_id);
}


__global__ void init_indices_kernel(int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        indices[idx] = idx;
    }
}

extern "C" void cuda_postprocess(float*& d_input, float*& d_output, int img0_w, int img0_h, cudaStream_t& stream)
{
    constexpr int threads = 256;
    int blocks = (NUM_BOXES + threads - 1) / threads;
    init_indices_kernel<<<blocks, threads, 0 , stream>>>(d_indices, NUM_BOXES);

    int blocks1 = (NUM_BOXES + threads - 1) / threads;
    argmax_kernel<<<blocks1, threads, 0, stream>>>(d_input, d_max_scores, d_max_class_ids);

    // CUB Top-K Sort using pre-allocated temp buffer
    cub::DeviceRadixSort::SortPairsDescending(
        d_sort_temp, d_sort_temp_bytes,
        d_max_scores, d_sorted_scores,
        d_indices, d_sorted_indices,
        NUM_BOXES, 0, sizeof(float) * 8, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float gain = std::min(static_cast<float>(kInputW) / static_cast<float>(img0_w),
                          static_cast<float>(kInputH) / static_cast<float>(img0_h));
    float ow = std::round(static_cast<float>(img0_w) * gain);
    float oh = std::round(static_cast<float>(img0_h) * gain);
    float pad_x = std::round((static_cast<float>(kInputW) - ow) / 2.0f - 0.1f);
    float pad_y = std::round((static_cast<float>(kInputH) - oh) / 2.0f - 0.1f);

    int out_blocks = (TOP_K + threads - 1) / threads;
    gather_and_format_kernel<<<out_blocks, threads, 0, stream>>>(
        d_input, d_sorted_indices, d_sorted_scores,
        d_max_class_ids, d_output,
        img0_w, img0_h, pad_x, pad_y, gain);
}

void cuda_postprocess_init() {
    CUDA_CHECK(cudaMalloc(&d_max_scores,      NUM_BOXES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_class_ids,   NUM_BOXES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices,         NUM_BOXES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_scores,   NUM_BOXES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sorted_indices,  NUM_BOXES * sizeof(int)));

    // Pre-allocate CUB sort temporary storage
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, d_sort_temp_bytes,
        static_cast<float*>(nullptr), static_cast<float*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        NUM_BOXES, 0, sizeof(float) * 8);
    // Add 50% safety margin for runtime variations
    d_sort_temp_bytes = d_sort_temp_bytes * 3 / 2;
    CUDA_CHECK(cudaMalloc(&d_sort_temp, d_sort_temp_bytes));
    std::cout << "CUB sort temp buffer: " << d_sort_temp_bytes << " bytes" << std::endl;
}

void cuda_postprocess_destroy() {
    if(d_max_scores)      CUDA_CHECK(cudaFree(d_max_scores));
    if(d_max_class_ids)   CUDA_CHECK(cudaFree(d_max_class_ids));
    if(d_indices)         CUDA_CHECK(cudaFree(d_indices));
    if(d_sorted_scores)   CUDA_CHECK(cudaFree(d_sorted_scores));
    if(d_sorted_indices)  CUDA_CHECK(cudaFree(d_sorted_indices));
    if(d_sort_temp)       CUDA_CHECK(cudaFree(d_sort_temp));

    d_max_scores = nullptr;
    d_max_class_ids = nullptr;
    d_indices = nullptr;
    d_sorted_scores = nullptr;
    d_sorted_indices = nullptr;
    d_sort_temp = nullptr;
}
