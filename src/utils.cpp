/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File utils.cpp
* Description: handle file operations
*/
#include "utils.h"
#include <bits/stdint-uintn.h>
#include <cstdint>
#include <map>
#include <iostream>
#include <fstream>
#include <cmath>
#include <unistd.h>
#include <cstring>
#include <dirent.h>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <regex>
#include "acl/acl.h"

using namespace std;

aclrtRunMode Utils::runMode_ = ACL_DEVICE;
/*const static std::vector<std::string> Label = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};    */
const static std::vector<std::string> Label = {"warplane"};
const static uint32_t kGridSize[3][2] = {{40, 80}, {20, 40}, {10, 20}};
// const uint numBBoxes = 3;
const uint numClasses = 1;
// const uint BoxTensorLabel = 12;
const float nmsThresh = 0.45;
const float MaxBoxClassThresh = 0.3;
const int anchor[3][3][2] = {{{10,13},{16,30},{33,23}}, {{30,61},{62,45},{59,119}}, {{116,90},{156,198},{373,326}}};
const int stride[3] = {8, 16, 32};

const uint32_t kLineSolid = 2;
const double kFountScale = 0.5;
const cv::Scalar kFontColor(0,0,255);
const uint32_t kLabelOffset = 11;
const vector<cv::Scalar> kColors{
    cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255),
    cv::Scalar(50, 205, 50), cv::Scalar(139, 85, 26)};

const uint32_t kBBoxDataBufId = 0;
const uint32_t kBoxNumDataBufId = 1;
enum BBoxIndex { TOPLEFTX = 0, TOPLEFTY, BOTTOMRIGHTX, BOTTOMRIGHTY, SCORE, LABEL };

bool Utils::IsDirectory(const string &path) {
  // get path stat
  struct stat buf;
  if (stat(path.c_str(), &buf) != kStatSuccess) {
    return false;
  }

  // check
  if (S_ISDIR(buf.st_mode)) {
    return true;
  } else {
    return false;
  }
}

bool Utils::IsPathExist(const string &path) {
  ifstream file(path);
  if (!file) {
    return false;
  }
  return true;
}

void Utils::SplitPath(const string &path, vector<string> &path_vec) {
    char *char_path = const_cast<char*>(path.c_str());
    const char *char_split = kImagePathSeparator.c_str();
    char *tmp_path = strtok(char_path, char_split);
    while (tmp_path) {
        path_vec.emplace_back(tmp_path);
        tmp_path = strtok(nullptr, char_split);
    }
}

bool Utils::NaturalCompare(const std::string &a, const std::string &b) {
    std::regex number_re("(\\d+)");
    std::sregex_token_iterator it_a(a.begin(), a.end(), number_re, 0);
    std::sregex_token_iterator it_b(b.begin(), b.end(), number_re, 0);
    std::sregex_token_iterator end;

    while (it_a != end && it_b != end) {
        int num_a = std::stoi(*it_a++);
        int num_b = std::stoi(*it_b++);
        if (num_a != num_b)
            return num_a < num_b;
    }
    return a < b;
}

void Utils::GetAllFiles(const string &path, vector<string> &file_vec) {
    // split file path
    vector<string> path_vector;
    SplitPath(path, path_vector);

    for (string every_path : path_vector) {
        // check path exist or not
        if (!IsPathExist(path)) {
        ERROR_LOG("Failed to deal path=%s. Reason: not exist or can not access.",
                every_path.c_str());
        continue;
        }
        // get files in path and sub-path
        GetPathFiles(every_path, file_vec);
    }
    std::sort(file_vec.begin(), file_vec.end(), NaturalCompare);
}

void Utils::GetPathFiles(const string &path, vector<string> &file_vec) {
    struct dirent *dirent_ptr = nullptr;
    DIR *dir = nullptr;
    if (IsDirectory(path)) {
        dir = opendir(path.c_str());
        while ((dirent_ptr = readdir(dir)) != nullptr) {
            // skip . and ..
            if (dirent_ptr->d_name[0] == '.') {
            continue;
            }

            // file path
            string full_path = path + kPathSeparator + dirent_ptr->d_name;
            // directory need recursion
            if (IsDirectory(full_path)) {
                GetPathFiles(full_path, file_vec);
            } else {
                // put file
                file_vec.emplace_back(full_path);
            }
        }
    } 
    else {
        file_vec.emplace_back(path);
    }
}

void Utils::ImageNchw(shared_ptr<ImageDesc>& imageData, std::vector<cv::Mat>& nhwcImageChs, uint32_t size) {
    uint8_t* nchwBuf = new uint8_t[size];
    int channelSize = IMAGE_CHAN_SIZE_F32(nhwcImageChs[0].rows, nhwcImageChs[0].cols);
    int pos = 0;
    for (int i = 0; i < nhwcImageChs.size(); i++) {
        memcpy(static_cast<uint8_t *>(nchwBuf) + pos,  nhwcImageChs[i].ptr<float>(0), channelSize);
        pos += channelSize;
    }

    imageData->size = size;
    imageData->data.reset((uint8_t *)nchwBuf, [](uint8_t* p) { delete[](p);} );
}


vector<BBox> Utils::nmsAllClasses(const float nmsThresh, std::vector<BBox>& binfo, const uint numClasses)
{
    std::vector<BBox> result;
    std::vector<std::vector<BBox>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.cls).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }

    return result;
}

vector<BBox> Utils::nonMaximumSuppression(const float nmsThresh, std::vector<BBox> binfo){
    /* 非极大值抑制，去除重叠较大的框，只保留得分较高，代表性的目标框 */

    // 计算在一维上的重叠部分
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        float left = max(x1min, x2min);
        float right = min(x1max, x2max);
        return right-left;
    };

    // 
    auto computeIoU =[&overlap1D](BBox& bbox1, BBox& bbox2) -> float {
        float overlapX = overlap1D(bbox1.rect.ltX, bbox1.rect.rbX, bbox2.rect.ltX, bbox2.rect.rbX);
        float overlapY = overlap1D(bbox1.rect.ltY, bbox1.rect.rbY, bbox2.rect.ltY, bbox2.rect.rbY);
        if(overlapX <= 0 or overlapY <= 0) return 0;
        float area1 = (bbox1.rect.rbX - bbox1.rect.ltX) * (bbox1.rect.rbY - bbox1.rect.ltY);
        float area2 = (bbox2.rect.rbX - bbox2.rect.ltX) * (bbox2.rect.rbY - bbox2.rect.ltY);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(), 
        [](const BBox& b1, const BBox& b2) { return b1.score > b2.score;});
    std::vector<BBox> out;

    for (auto& i : binfo)
    {
        bool keep = true;
        for (auto& j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

void Utils::BufferDeleter(void *p) {
    if (!RunStatus::GetDeviceStatus()) {
        if (p != nullptr) {
            (void) aclrtFreeHost(p);
        }
    }
}

/* 画框 */
void Utils::DrawBoxToResult(const vector<BBox> &result, cv::Mat& imageInput) {
    for (size_t i = 0; i < result.size(); ++i) {
        BBox box = result[i];
        string cls_name = Label[box.cls];
        
        cv::Point p1, p2;
        p1.x = result[i].rect.ltX;
        p1.y = result[i].rect.ltY;
        p2.x = result[i].rect.rbX;
        p2.y = result[i].rect.rbY;
        
        // 置信度 ： std::to_string(box.score) + "\%"
        cv::putText(imageInput, cls_name, p1, 0, 1, kColors[i % kColors.size()], 2, 4, 0);
        cv::rectangle(imageInput, p1, p2, kColors[i % kColors.size()], kLineSolid);
    }
}
