#include "preprocess.cuh"

uint8_t* img_buffer_host    = nullptr;      // preprocess
uint8_t* img_buffer_device  = nullptr;      // preprocess

__global__ void warpaffine_kernel_2d(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int dst_w, int dst_h,
    int src_w, int src_h,
    int src_step,
    float pad_value,
    int new_unpad_w, int new_unpad_h,
    float left, float top,
    float scale_x, float scale_y,
    float norm_factor)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    int plane_size = dst_w * dst_h;
    int pixel_idx = dy * dst_w + dx;

    float fx = dx - left;
    float fy = dy - top;

    if (fx < 0 || fx >= new_unpad_w || fy < 0 || fy >= new_unpad_h) {
        // Padding no img area
        dst[0 * plane_size + pixel_idx] = pad_value;
        dst[1 * plane_size + pixel_idx] = pad_value;
        dst[2 * plane_size + pixel_idx] = pad_value;
        return;
    }

    // cv2.resize formula:
    // src_coord = (dst_coord + 0.5) * (src_size / dst_size) - 0.5
    float sx = (fx + 0.5f) * scale_x - 0.5f;
    float sy = (fy + 0.5f) * scale_y - 0.5f;

    // bilinear interpolation
    int x0 = __float2int_rd(sx);
    int y0 = __float2int_rd(sy);
    float wx = sx - x0;
    float wy = sy - y0;

    // border clamp
    x0 = max(0, min(x0, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float w00 = (1.0f - wx) * (1.0f - wy);  // left  top
    float w10 = wx * (1.0f - wy);           // right top
    float w01 = (1.0f - wx) * wy;           // left  bottom
    float w11 = wx * wy;                    // right bottom

    const uchar3* src3 = reinterpret_cast<const uchar3*>(src);
    int src_width_in_uchar3 = src_step / 3;
    uchar3 c00 = src3[y0 * src_width_in_uchar3 + x0];
    uchar3 c10 = src3[y0 * src_width_in_uchar3 + x1];
    uchar3 c01 = src3[y1 * src_width_in_uchar3 + x0];
    uchar3 c11 = src3[y1 * src_width_in_uchar3 + x1];

    float val_b = w00 * c00.x + w10 * c10.x + w01 * c01.x + w11 * c11.x;
    float val_g = w00 * c00.y + w10 * c10.y + w01 * c01.y + w11 * c11.y;
    float val_r = w00 * c00.z + w10 * c10.z + w01 * c01.z + w11 * c11.z;

    // BGR->RGB -> norm -> rgbrgbrgb to rrrgggbbb
    dst[0 * plane_size + pixel_idx] = val_r * norm_factor;  // R plane
    dst[1 * plane_size + pixel_idx] = val_g * norm_factor;  // G plane
    dst[2 * plane_size + pixel_idx] = val_b * norm_factor;  // B plane

}


extern "C" void cuda_preprocess(cv::Mat& img, float*& dst, cudaStream_t& stream) {
    int src_w = img.cols;
    int src_h = img.rows;
    size_t src_step = img.step[0];  // w * 4
    size_t img_size = src_step * src_h;

    //memcpy(img_buffer_host, img.ptr(), img_size);
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device,img.ptr(), img_size, cudaMemcpyHostToDevice, stream));

    float r = std::min(static_cast<float>(kInputH) / src_h,  static_cast<float>(kInputW) / src_w);
    int new_unpad_w = static_cast<int>(std::round(src_w * r));
    int new_unpad_h = static_cast<int>(std::round(src_h * r));
    float dw = (kInputW - new_unpad_w) / 2.0f;
    float dh = (kInputH - new_unpad_h) / 2.0f;
    float left = std::round(dw - 0.1f);
    float top  = std::round(dh - 0.1f);

    float scale_x = static_cast<float>(src_w) / new_unpad_w;
    float scale_y = static_cast<float>(src_h) / new_unpad_h;
    constexpr float norm_factor = 1.0f / 255.0f;
    float pad_value = 114.0f * norm_factor;

    dim3 block(16, 16, 1);
    dim3 grid(
        (kInputW + block.x - 1) / block.x,
        (kInputH+ block.y - 1) / block.y,
        1
    );

    warpaffine_kernel_2d<<<grid, block, 0, stream>>>(
        img_buffer_device, dst,
        kInputW, kInputH,
        src_w, src_h,
        src_step,
        pad_value,
        new_unpad_w,new_unpad_h,
        left, top,
        scale_x, scale_y,
        norm_factor);

    //CUDA_CHECK(cudaGetLastError());
}

void cuda_preprocess_init() {
    // prepare input data in pinned memory
    CUDA_CHECK(cudaMallocHost((void **) &img_buffer_host, kMaxInputImageSize));
    // prepare input data in device memory
    CUDA_CHECK(cudaMalloc((void **) &img_buffer_device, kMaxInputImageSize));
}

void cuda_preprocess_destroy() {
    if (img_buffer_host) {
        CUDA_CHECK(cudaFreeHost(img_buffer_host));
        img_buffer_host = nullptr;
    }
    if (img_buffer_device) {
        CUDA_CHECK(cudaFree(img_buffer_device));
        img_buffer_device = nullptr;
    }
}
