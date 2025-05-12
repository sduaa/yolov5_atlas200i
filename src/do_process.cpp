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

* File do_process.cpp
* Description: handle acl resource
*/
#include "do_process.h"
#include <iostream>
#include "model_process.h"
#include "acl/acl.h"
#include "utils.h"
#include <time.h>

using namespace std;
bool RunStatus::isDevice_ = false;

DoProcess::DoProcess(const char* modelPath, uint32_t modelWidth, uint32_t modelHeight)
/* 初始化成员列表，对类成员赋初值 */
    :deviceId_(0), context_(nullptr), stream_(nullptr), inputBuf_(nullptr),
    modelWidth_(modelWidth), modelHeight_(modelHeight), isInited_(false),isDeviceSet_(false), 
    Label({"warplane"})
{
    modelPath_ = modelPath;
    inputDataSize_ = RGBU8_IMAGE_SIZE(modelWidth_, modelHeight_);
}

DoProcess::~DoProcess()
{
    DestroyResource();
}


Result DoProcess::InitResource()
{
    const char *aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }

    INFO_LOG("acl init success");

    // open device
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }

    return SUCCESS;
}

Result DoProcess::InitModel(const char* omModelPath) {
    /* 加载OM模型并初始化推理所需的输入输出资源*/
    Result ret = model_.LoadModelFromFileWithMem(omModelPath);   // 加载模型到内存
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = model_.CreateDesc();      // 创建模型描述符
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = model_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    aclrtMalloc(&inputBuf_, (size_t)(inputDataSize_), ACL_MEM_MALLOC_HUGE_FIRST);
    if (inputBuf_ == nullptr) {
        ERROR_LOG("Acl malloc image buffer failed.");
        return FAILED;
    }

    ret = model_.CreateInput(inputBuf_, inputDataSize_);
    if (ret != SUCCESS) {
        ERROR_LOG("Create mode input dataset failed");
        return FAILED;
    }

    return SUCCESS;
}

Result DoProcess::Init() {
    /* 类的初始化函数，初始化推理中所需的资源 */
    if (isInited_) {
        INFO_LOG("Classify instance is initied already!");
        return SUCCESS;
    }

    Result ret = InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("Init acl resource failed");
        return FAILED;
    }

    ret = InitModel(modelPath_);
    if (ret != SUCCESS) {
        ERROR_LOG("Init model failed");
        return FAILED;
    }

    isInited_ = true;
    return SUCCESS;
}


Result DoProcess::PrePreprocess(const string& imageFile) {
    /* 读取并预处理，转换为模型需要的格式，并将图像数据复制到推理使用的内存中*/

    clock_t start, finish;
    double duration = 0.0;
    start = clock();

    INFO_LOG("Read image %s", imageFile.c_str());
    cv::Mat origMat = cv::imread(imageFile, 1);   //BGR
    if (origMat.empty()) {
        ERROR_LOG("Read image failed");
        return FAILED;
    }

    //resize
    cv::Mat resizeMat;
    cv::resize(origMat, resizeMat, cv::Size(modelWidth_, modelHeight_));
    if (resizeMat.empty()) {
        ERROR_LOG("Resize image failed");
        return FAILED;
    }

    cv::Mat inputMat;
    cv::cvtColor(resizeMat, inputMat, cv::COLOR_BGR2RGB);

    if (runMode_ == ACL_HOST) {
        //Atals200DKi上运行时,需要将图片数据拷贝到device侧
        aclError ret = aclrtMemcpy(inputBuf_, inputDataSize_,inputMat.ptr<uint8_t>(), inputDataSize_,ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("Copy resized image data to device failed.");
            return FAILED;
        }
    } else {
        //Atals200DK上运行时,数据拷贝到本地即可.
        //reiszeMat是局部变量,数据无法传出函数,需要拷贝一份
        memcpy(inputBuf_, inputMat.ptr<void>(), inputDataSize_);
    }

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC * 1000;
    INFO_LOG("preprocss cost: %.0f ms", duration);

    return SUCCESS;
}

Result DoProcess::Inference(aclmdlDataset*& inferenceOutput) {
    /* 推理，并返回推理输出数据的指针 */

    clock_t start, finish;
    double duration = 0.0;
    start = clock();

    Result ret = model_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC * 1000;
    INFO_LOG("inference cost: %.0f ms", duration);

    inferenceOutput = model_.GetModelOutputData();

    return SUCCESS;
}

shared_ptr<void> DoProcess::GetInferenceOutputItem(aclmdlDataset* inferenceOutput){
    /* 将输出从 device 拷贝回 host */

    shared_ptr<void> dataBuff = nullptr;
    
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(inferenceOutput, 0);
    void *data = aclGetDataBufferAddr(dataBuffer);
    size_t len = aclGetDataBufferSizeV2(dataBuffer);

    if (RunStatus::GetDeviceStatus()) {
        shared_ptr<void> ptr(data, Utils::BufferDeleter);
        dataBuff = ptr;
    } else {
        void *outHostData = nullptr;
        aclError ret = aclrtMallocHost(&outHostData, len);
        if (ret != ACL_ERROR_NONE) {
            cout << "aclrtMallocHost failed, result code is " << ret << endl;
            return dataBuff;
        }
        ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            cout << "aclrtMemcpy failed, result code is " << ret << endl;
            (void) aclrtFree(outHostData);
            return dataBuff;
        }
        shared_ptr<void> ptr(outHostData, Utils::BufferDeleter);
        dataBuff = ptr;
    }
    INFO_LOG("model output get success");
    return dataBuff;
}

std::vector<BBox> DoProcess::PostProcess(const string &path, aclmdlDataset *modelOutput)
{
    /* 后处理，找到框和类别 */

    vector<BBox> bboxesNew;
    uint32_t dataSize = 0;
    shared_ptr<void> output = GetInferenceOutputItem(modelOutput);
    float* detectData = static_cast<float*>(output.get()); 

    uint32_t totalBox = 25200;   //FIXME
    vector<BBox> detectResults;

    int H,W;
    H = 1000;
    W = 1000;

    float widthScale = float(W) / float(modelWidth_);
    float heightScale = float(H) / float(modelHeight_);

    clock_t start, finish;
    double duration = 0.0;
    start = clock();

    for (uint32_t i = 0; i < totalBox; i++) {
        BBox boundBox;

        float score = (detectData[i*6 + 4]);
        if (score < 0.3){
            continue;
        }

        int center_x = detectData[i*6 + 0] * widthScale;
        int center_y = detectData[i*6 + 1] * heightScale;
        int width = detectData[i*6 + 2] * widthScale;
        int height = detectData[i*6 +  3] * heightScale;

        boundBox.rect.ltX = std::max(0, center_x - width / 2);
        boundBox.rect.ltY = std::max(0, center_y - height / 2);
        boundBox.rect.rbX = std::min(W, center_x + width / 2);
        boundBox.rect.rbY = std::min(H, center_y + height / 2);
	    boundBox.cls = 0;//FIXME
        detectResults.emplace_back(boundBox);
    }
    
/*
#pragma omp parallel
    {
        std::vector<BBox> localResults;

        #pragma omp for nowait
        for (uint32_t i = 0; i < totalBox; i++) {
            float score = detectData[i*6 + 4];
            if (score < 0.3) continue;

            int center_x = detectData[i*6 + 0] * widthScale;
            int center_y = detectData[i*6 + 1] * heightScale;
            int width = detectData[i*6 + 2] * widthScale;
            int height = detectData[i*6 + 3] * heightScale;

            BBox boundBox;
            boundBox.rect.ltX = std::max(0, center_x - width / 2);
            boundBox.rect.ltY = std::max(0, center_y - height / 2);
            boundBox.rect.rbX = std::min(W, center_x + width / 2);
            boundBox.rect.rbY = std::min(H, center_y + height / 2);
            boundBox.cls = 0; // FIXME

            localResults.emplace_back(boundBox);
        }

        #pragma omp critical
        detectResults.insert(detectResults.end(), localResults.begin(), localResults.end());
    }
    */

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC * 1000;
    INFO_LOG("postprocess cost: %.0f ms", duration);
    
    if (runMode_ == ACL_HOST) {
        delete[]((uint8_t *)detectData);
    }

    //NMS
    bboxesNew = Utils::nmsAllClasses(nmsThresh, detectResults, numClasses);

    return bboxesNew;
}

void DoProcess::DestroyResource()
{
    // clear resources.
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId_);

    //TODO:
    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");

}
