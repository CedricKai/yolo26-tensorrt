#include "utils.h"

void print_gpu_info(int deviceId)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    std::cout << "====== GPU Info ======\n";
    std::cout << "GPU Name           : " << prop.name << "\n";
    std::cout << "Global Memory      : " << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024 * 1024) << " GB\n";
    std::cout << "Compute Capability : " << prop.major << "." << prop.minor << "\n";
    std::cout << "======================\n";
}

bool read_files_in_dir(const char* dir_path, std::vector<std::string>& file_names)
{
    DIR* p_dir = opendir(dir_path);
    if (p_dir == nullptr) {
        std::cerr << "open directory failed: " << dir_path << std::endl;
        return false;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr)
    {
        if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0)
        {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return true;
}

bool load_class_names(const std::string& label_path, std::vector<std::string>& names)
{
    names.clear();
    std::ifstream ifs(label_path);
    if (!ifs.is_open())
    {
        std::cerr << "open label file failed: " << label_path << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.find_first_not_of(" \t") == std::string::npos)
            continue;

        std::string name;
        auto pos = line.find(':');
        if (pos != std::string::npos)
            name = line.substr(pos + 1);
        else
            name = line;
        size_t start = name.find_first_not_of(" \t");
        size_t end   = name.find_last_not_of(" \t");
        if (start != std::string::npos)
            name = name.substr(start, end - start + 1);
        else
            name.clear();
        if (!name.empty())
            names.push_back(name);
    }
    return true;
}

static const std::vector<cv::Scalar> BBOX_COLORS = {
    {0,   0,   255},        // 纯红
    {0,  255,   0},         // 纯绿
    {255,  0,    0},        // 纯蓝
    {0, 255,  255},         // 黄色
    {255, 255,   0},        // 青色
    {255,  0,  255},        // 品红
    {0,  80,  255},         // 橙红
    {30, 144, 255},         // 道奇蓝
    {130,  0,  75},         // 深紫
    {0, 165, 255},          // 橙色
    {127, 255,   0},        // 春绿
    {255, 144,  30},        // 深天蓝
    {0,   0,  180},         // 深红
    {180,   0,    0},       // 深蓝
    {0, 180, 180},          // 暗黄
    {180,  0, 180}          // 暗品红
};

void draw_bbox(cv::Mat& img,
    const std::vector<Detection>& detections,
    const std::vector<std::string>& class_names) {

    int draw_count = std::min(static_cast<int>(detections.size()), kVisMax);
    if (draw_count == 0 || img.empty()) {
        std::cout << "[DrawBBox] Warning: No valid bboxes to draw or image is empty." << std::endl;
        return;
    }

    int shape_sum = img.cols + img.rows + img.channels();
    int line_width = std::max(static_cast<int>(std::round(shape_sum * 0.0015)), 2);
    float font_scale = line_width / 3.5f;
    std::cout << std::left << std::setw(20) << "Detected objects: " << draw_count << std::endl;
    for (int i = 0; i < draw_count; ++i) {
        const auto& det = detections[i];

        int x1 = std::max(0, static_cast<int>(det.left));
        int y1 = std::max(0, static_cast<int>(det.top));
        int x2 = std::min(img.cols - 1, static_cast<int>(det.right));
        int y2 = std::min(img.rows - 1, static_cast<int>(det.bottom));

        int class_id = static_cast<int>(det.class_id);
        std::string label_name = (class_id >= 0 && class_id < static_cast<int>(class_names.size()))
            ? class_names[class_id] : ("id:" + std::to_string(class_id));

        std::cout << std::left << std::setw(10) << label_name << "   [ "
                  << std::fixed << std::setprecision(3)
                  << std::setw(8) << det.left  << ", "
                  << std::setw(8) << det.top   << ", "
                  << std::setw(8) << det.right << ", "
                  << std::setw(8) << det.bottom << ",   "
                  << std::setw(6) << det.conf  << ",   "
                  << std::defaultfloat
                  << class_id
                  << " ]" << '\n';

        int color_idx = class_id % static_cast<int>(BBOX_COLORS.size());
        if (color_idx < 0) color_idx = 0;
        cv::Scalar color = BBOX_COLORS[color_idx];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, line_width);

        std::ostringstream oss;
        oss << label_name;
        oss << " " << std::fixed << std::setprecision(2) << det.conf;
        std::string label = oss.str();

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, line_width, &baseline);
        int tw = text_size.width;
        int th = text_size.height;

        cv::rectangle(img,
                      cv::Point(x1 - 2, y1 - th - baseline - 2),
                      cv::Point(x1 + tw, y1),
                      color,
                      cv::FILLED);
        cv::putText(img,
                    label,
                    cv::Point(x1, y1 - baseline),
                    cv::FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    cv::Scalar(255, 255, 255),
                    line_width);
    }
}

