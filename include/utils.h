#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include "config.h"


void print_gpu_info(int deviceId=0);

bool load_class_names(const std::string& label_path, std::vector<std::string>& names);

bool read_files_in_dir(const char* dir_path, std::vector<std::string>& file_names);

void draw_bbox(cv::Mat& img, const std::vector<Detection>& detections, const std::vector<std::string>& class_names);