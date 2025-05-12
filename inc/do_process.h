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

* File sample_process.h
* Description: handle acl resource
*/
#pragma once
#include "utils.h"
#include "acl/acl.h"
#include "model_process.h"
#include <memory>

template<class Type>
std::shared_ptr<Type> MakeSharedNoThrow() {
  try {
    return std::make_shared<Type>();
  } catch (...) {
    return nullptr;
  }
}

#define MAKE_SHARED_NO_THROW(memory, memory_type) \
    memory = MakeSharedNoThrow<memory_type>();

/**
* DoProcess
*/
class DoProcess {
public:
    /**
    * @brief Constructor
    */
    DoProcess(const char* modelPath, uint32_t modelWidth, uint32_t modelHeight);

    /**
    * @brief Destructor
    */
    ~DoProcess();

    /**
    * @brief init reousce
    * @return result
    */
    Result InitResource();
    Result InitModel(const char* omModelPath);
    Result Init();
    
    /**
    * @brief encode sample process
    * @param [in] input_path: input image path
    * @return result
    */
    Result PrePreprocess(const string& imageFile);

    Result Inference(aclmdlDataset*& inferenceOutput);

    shared_ptr<void>  GetInferenceOutputItem(aclmdlDataset* inferenceOutput);

    std::vector<BBox> PostProcess(const string &path, aclmdlDataset *modelOutput);

    void DestroyResource();

private:
    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
    ModelProcess model_;

    const char* modelPath_;
    uint32_t modelWidth_;
    uint32_t modelHeight_;
    uint32_t inputDataSize_;
    void*    inputBuf_;

    aclrtRunMode runMode_;

    bool isInited_;
    bool isDeviceSet_;
    cv::Mat origImage;

    enum BBoxIndex { TOPLEFTX = 0, TOPLEFTY, BOTTOMRIGHTX, BOTTOMRIGHTY, SCORE, LABEL };
    const float nmsThresh = 0.45;
    const uint numClasses = 1;
    const std::vector<std::string> Label;
};

