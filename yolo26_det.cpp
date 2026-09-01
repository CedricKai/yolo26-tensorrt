#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "logging.h"
#include "model.h"
#include "config.h"
#include "cuda_utils.h"
#include "preprocess.cuh"
#include "postprocess.cuh"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;

float gpu_ms;
cudaEvent_t g_ev_start;
cudaEvent_t g_ev_stop;

void init_events()
{
    CUDA_CHECK(cudaEventCreate(&g_ev_start));
    CUDA_CHECK(cudaEventCreate(&g_ev_stop));
}
void release_events()
{
    CUDA_CHECK(cudaEventDestroy(g_ev_start));
    CUDA_CHECK(cudaEventDestroy(g_ev_stop));
}

inline PrecisionMode string_to_precision(const std::string& s)
{
    if(s == "fp16") return PrecisionMode::FP16;
    if(s == "int8") return PrecisionMode::INT8;
    return PrecisionMode::FP32;
}

inline void setTrtConfig(IBuilder* builder, IBuilderConfig* config, const PrecisionMode& mode) {
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 256U * (1U << 20));
    switch(mode)
    {
        case PrecisionMode::FP16:
            config->setFlag(BuilderFlag::kFP16);
            config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
            std::cout << "Inference precision configured as FP16." << std::endl;
            break;
        case PrecisionMode::INT8:
            config->setFlag(BuilderFlag::kINT8);
            std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
            assert(builder->platformHasFastInt8());
            config->setFlag(BuilderFlag::kINT8);
            // auto* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, kInputQuantizationFolder, "int8calib.table",
            //                                               kInputTensorName);
            // config->setInt8Calibrator(calibrator);
            std::cout << "Inference precision configured as INT8." << std::endl;
            break;
        case PrecisionMode::FP32:
            config->clearFlag(BuilderFlag::kTF32);
            config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
            std::cout << "Inference precision configured as FP32." << std::endl;
            break;
        default:
            break;
    }
}


void serialize_engine(std::string& wts_name,
    std::string& engine_name,
    float& gd, float& gw, int& max_channels, std::string& model_scale, std::string& dtype) {
    // read from .wts
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    IBuilderConfig* config = builder->createBuilderConfig();
    setTrtConfig(builder, config, string_to_precision(dtype));
    IHostMemory*  serialized_engine = build_yolo26_engine(builder, config, DataType::kFLOAT, wts_name,
        gd, gw, max_channels, model_scale);
    assert(serialized_engine);

    // write to .engine
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    delete serialized_engine;
    delete config;
    delete builder;
}


bool parse_args(int argc, char** argv,
    std::string& wts, std::string& engine, std::string& img_dir,
    std::string& model_scale, std::string& device, std::string& mode, std::string& dtype,
    float& gd, float& gw, int& max_channels) {
    if (argc < 5)
        return false;

    if (std::string(argv[1]) == "-s") {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        model_scale = std::string(argv[4]);
        if (model_scale[0] == 'n') {
            gd = 0.5;
            gw = 0.25;
            max_channels = 1024;
        } else if (model_scale[0] == 's') {
            gd = 0.5;
            gw = 0.5;
            max_channels = 1024;
        } else if (model_scale[0] == 'm') {
            gd = 0.5;
            gw = 1.0;
            max_channels = 512;
        } else if (model_scale[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
        } else if (model_scale[0] == 'x') {
            gd = 1.0;
            gw = 1.5;
            max_channels = 512;
        } else {
            return false;
        }
        dtype= std::string(argv[5]);
    } else if (std::string(argv[1]) == "-d") {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        mode = std::string(argv[4]);
        if(mode != "p" && mode != "s" && mode != "c")
        {
            std::cerr << "mode must be p / s / c" << std::endl;
            return false;
        }
    } else {
        return false;
    }
    return true;
}


void deserialize_engine(std::string& engine_name,
    IRuntime** runtime,
    ICudaEngine** engine,
    IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    auto serialized_engine = std::make_unique<char[]>(size);
    assert(serialized_engine);
    file.read(serialized_engine.get(), size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine.get(), size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
}

void prepare_buffer(ICudaEngine* engine,
    const std::string& device,
    float** input_buffer_host,
    float** input_buffer_device,
    float** output_buffer_host,
    float** output_buffer_device,
    float** decode_ptr_host,
    float** decode_ptr_device) {

    assert(engine->getNbIOTensors() == 2);
    assert(engine->getTensorIOMode(kInputTensorName)  == nvinfer1::TensorIOMode::kINPUT);
    assert(engine->getTensorIOMode(kOutputTensorName) == nvinfer1::TensorIOMode::kOUTPUT);

    cuda_preprocess_init();
    cuda_postprocess_init();

    auto inputDims = engine->getTensorShape(kInputTensorName);
    auto outputDims = engine->getTensorShape(kOutputTensorName);
    int64_t i_vol = 1;
    int64_t o_vol = 1;
    for(int i = 0; i < inputDims.nbDims; i++) i_vol *= inputDims.d[i];
    for(int i = 0; i < outputDims.nbDims; i++) o_vol *= outputDims.d[i];
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, i_vol * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, o_vol * sizeof(float)));

    *decode_ptr_host = new float[kMaxNumOutputBboxSIZE];
    CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, kMaxNumOutputBboxSIZE * sizeof(float)));
}


void preprocess(cv::Mat& img, float*& dst, cudaStream_t& stream){
    cudaEventRecord(g_ev_start, stream);
    cuda_preprocess(img, dst, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEventRecord(g_ev_stop, stream);
    cudaEventSynchronize(g_ev_stop);
    cudaEventElapsedTime(&gpu_ms, g_ev_start, g_ev_stop);
    std::cout << std::left  << std::setw(20) << "Preprocess time:" << gpu_ms * 1e3  << " us\n";
}


void infer(IExecutionContext& context, cudaStream_t& stream) {
    // inference
    cudaEventRecord(g_ev_start, stream);
    context.enqueueV3(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEventRecord(g_ev_stop, stream);
    cudaEventSynchronize(g_ev_stop);
    cudaEventElapsedTime(&gpu_ms, g_ev_start, g_ev_stop);
    std::cout << std::left  << std::setw(20) << "Inference time:" << gpu_ms * 1e3 << " us\n";
}

void postprocess(float*&output, float*& dp_device, float*& dp_host, std::vector<Detection>& res,
    int src_w, int src_h, cudaStream_t& stream) {
    cudaEventRecord(g_ev_start, stream);
    cuda_postprocess(output, dp_device, src_w, src_h, stream);
    CUDA_CHECK(cudaMemcpyAsync(dp_host, dp_device,kMaxNumOutputBboxSIZE * sizeof(float),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEventRecord(g_ev_stop, stream);
    cudaEventSynchronize(g_ev_stop);
    cudaEventElapsedTime(&gpu_ms, g_ev_start, g_ev_stop);

    res.clear();
    Detection* det_ptr = reinterpret_cast<Detection*>(dp_host);
    for(int i = 0; i < kMaxNumOutputBbox; i++)
    {
        Detection box = det_ptr[i];
        int class_id = static_cast<int>(box.class_id);
        if(class_id == -1)
        {
            break;
        }
        res.push_back(box);
    }
    std::cout << std::left  << std::setw(20) << "Postprocess time:" << gpu_ms * 1e3 << " us\n";
}

void run_one_frame(cv::Mat& frame,
                    IExecutionContext& context,
                    float* input_buffer_device,
                    float* output_buffer_device,
                    float* decode_ptr_device,
                    float* decode_ptr_host,
                    std::vector<Detection>& res,
                    std::vector<std::string>& class_names,
                    cudaStream_t& stream,
                    bool draw = true,
                    bool draw_fps = true)
{
    auto t0 = std::chrono::steady_clock::now();
    if (frame.empty()) return;
    preprocess(frame, input_buffer_device, stream);
    infer(context, stream);
    postprocess(output_buffer_device, decode_ptr_device, decode_ptr_host, res,
        frame.cols, frame.rows, stream);
    if (draw)
        draw_bbox(frame, res, class_names);
    if(draw_fps)
    {
        auto t1 = std::chrono::steady_clock::now();
        double duration_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::string fps_text = cv::format("FPS: %.1f", 1e6 / duration_us);
        cv::putText(frame, fps_text, cv::Point(15, 35),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 0),
                    2);
        std::cout << "Total speed time -> " << duration_us / 1e3  << " ms\n";
    }
}


int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    std::string wts_name = "";
    std::string engine_name = "";
    std::string input_path = "";
    std::string model_scale = "";
    std::string device = "";
    std::string mode = "";   // p / s / c
    std::string dtype = "";
    float gd = 0.0f, gw = 0.0f;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, input_path, model_scale, device, mode, dtype,
        gd, gw, max_channels)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolo26 -s [.wts] [.engine] [n/s/m/l/x] [fp32/fp16/int8] serialize engine\n";
        std::cerr << "./yolo26 -d [.engine] [input_path] [p/s/c]\n";
        std::cerr << "mode: p=image folder, s=single video, c=camera(0)\n";
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(wts_name, engine_name, gd, gw, max_channels, model_scale, dtype);
        return 0;
    }

    std::vector<std::string> class_names;
    if(!load_class_names("../file/coco.txt", class_names))
        return -1;

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Prepare cpu and gpu buffers
    float* input_buffer_host    = nullptr;      // inference input   no use
    float* input_buffer_device  = nullptr;      // inference input
    float* output_buffer_host   = nullptr;      // inference output  no use
    float* output_buffer_device = nullptr;      // inference output
    float* decode_ptr_host      = nullptr;      // postprocess
    float* decode_ptr_device    = nullptr;      // postprocess
    prepare_buffer(engine, device,
            &input_buffer_host, &input_buffer_device,
            &output_buffer_host, &output_buffer_device,
            &decode_ptr_host, &decode_ptr_device);
    context->setTensorAddress(kInputTensorName, input_buffer_device);       // input
    context->setTensorAddress(kOutputTensorName, output_buffer_device);     // output

    // gpu info
    print_gpu_info();
    init_events();

    // warmup
    std::cout << "\nWarm Up Start:" << std::endl;
    for (int i = 0; i < kWarmUp; i++)
    {
        auto start = std::chrono::steady_clock::now();
        cv::Mat img = cv::Mat::zeros(kInputH, kInputW, CV_8UC3);
        img = img.clone();
        std::vector<Detection> res;
        run_one_frame(img, *context, input_buffer_device, output_buffer_device, decode_ptr_device, decode_ptr_host,
            res, class_names, stream, false, false);
        auto end = std::chrono::steady_clock::now();
        double duration_us = std::chrono::duration<double, std::micro>(end - start).count();
        std::cout << "warm_up -> " << i << " time -> " << duration_us * 1000 << " us\n";
    }

    std::cout << "\nInference Start:" << std::endl;
    std::vector<Detection> res;
    if(mode == "p")
    {
        std::vector<std::string> file_names;
        if (!read_files_in_dir(input_path.c_str(), file_names))
            return -1;

        for (auto& file : file_names) {
            std::cout << "Running inference on image >>> " << file << "\n";
            cv::Mat img = cv::imread(input_path + "/" + file);
            if(img.empty()) {
                std::cerr << "read image failed:" << file << std::endl;
                continue;
            }
            img = img.clone();
            run_one_frame(img, *context, input_buffer_device, output_buffer_device,
                            decode_ptr_device, decode_ptr_host, res, class_names, stream, true, false);
            cv::imwrite("../output/trt_" + file, img);
        }
    }
    else if(mode == "s")
    {
        cv::VideoCapture cap(input_path);
        if(!cap.isOpened())
        {
            std::cerr << "open video failed: " << input_path << std::endl;
            return -1;
        }
        int fps = cap.get(cv::CAP_PROP_FPS);
        int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "Video fps:" << fps << " w:" << width << " h:" << height << std::endl;
        cv::VideoWriter writer;
        writer.open("../output/out_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width,height));
        cv::Mat frame;
        while (cap.read(frame))
        {
            run_one_frame(frame, *context, input_buffer_device, output_buffer_device,
                            decode_ptr_device, decode_ptr_host, res, class_names, stream);
            writer.write(frame);
            cv::imshow("video_detect", frame);
            if(cv::waitKey(1) == 27) break;
        }
        cap.release();
        writer.release();
        cv::destroyAllWindows();
    }
    else if(mode == "c")
    {
        cv::VideoCapture cap(0);
        if(!cap.isOpened())
        {
            std::cerr << "open camera 0 failed" << std::endl;
            return -1;
        }
        cv::Mat frame;
        while(true)
        {
            cap.read(frame);
            if(frame.empty()) break;
            run_one_frame(frame, *context, input_buffer_device, output_buffer_device,
                            decode_ptr_device, decode_ptr_host, res, class_names, stream);
            cv::imshow("camera_detect", frame);
            if(cv::waitKey(1) == 27) break; //ESC退出
        }
        cap.release();
        cv::destroyAllWindows();
    }

    // destroy
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(input_buffer_device));
    CUDA_CHECK(cudaFree(output_buffer_device));
    CUDA_CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    cuda_preprocess_destroy();
    cuda_postprocess_destroy();
    release_events();
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
