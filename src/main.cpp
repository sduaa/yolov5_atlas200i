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

* File main.cpp
* Description: dvpp yolo main func
*/

#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <sys/time.h>
#include "do_process.h"
#include "utils.h"

using namespace std;
namespace {
    uint32_t kModelWidth = 640;
    uint32_t kModelHeight = 640;
    const char* kModelPath = "../model/best_wsl.om";
}
    

int main(int argc, char *argv[])
{
    DoProcess process_yolo(kModelPath, kModelWidth, kModelHeight);
    Result ret = process_yolo.Init();
    if (ret != SUCCESS) {
        ERROR_LOG("process_yolo Init resource failed");
        return FAILED;
    }

    string input_path = "../data/event_frames";
    vector<string> fileVec;
    Utils::GetAllFiles(input_path, fileVec);
    if (fileVec.empty()) {
        ERROR_LOG("Failed to deal all empty path=%s.", input_path.c_str());
        return FAILED;
    }

    cv::namedWindow("Display Window", cv::WINDOW_NORMAL);
    cv::resizeWindow("Display Window", 1282, 640);

    clock_t start, finish;
    double duration = 0.0;
    INFO_LOG("please press Spacebar");
    while(cv::waitKey(1) != 32){};

    for (string imageFile : fileVec) {
        start = clock();

        Result ret = process_yolo.PrePreprocess(imageFile);
        if (ret != SUCCESS) {
            ERROR_LOG("Read file %s failed, continue to read next",
                      imageFile.c_str());                
            continue;
        }
        
        aclmdlDataset* inferenceOutput = nullptr;
        ret = process_yolo.Inference(inferenceOutput);
        if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return FAILED;
        }

        vector<BBox> boxlists = process_yolo.PostProcess(imageFile, inferenceOutput);
        if (boxlists.empty()) {
            ERROR_LOG("pull model output data failed");
            continue;
        }

        finish = clock();

        cv::Mat ori_img = cv::imread(imageFile);
        cv::Mat res_img = ori_img.clone();
        cv::Mat merged_img, res_border;
        Utils::DrawBoxToResult(boxlists, res_img);
        cv::resize(ori_img, ori_img, cv::Size(640, 640));
        cv::resize(res_img, res_img, cv::Size(640, 640));
        cv::copyMakeBorder(res_img, res_border, 
            0, 0, 2, 0,           // 上、下、左、右的边框宽度（左=2像素）
            cv::BORDER_CONSTANT, cv::Scalar(0, 0, 255)          // R
        );
        cv::hconcat(ori_img, res_border, merged_img);

        cv::imshow("Display Window", merged_img);
        cv::waitKey(10);

        duration = (double)(finish - start) / CLOCKS_PER_SEC * 1000;
        INFO_LOG("total cost: %.0f ms", duration);
    }

    INFO_LOG("execute yolo success");
    return SUCCESS;
}
