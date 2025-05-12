# YOLOV5_Ascend_CPP

该项目用于将 YOLOv5 模型部署到华为 Atlas 200i 平台，使用 C++ 实现推理逻辑。

## 模型转换

```bash
python export.py --include onnx --weights yolov5s.pt --img 640 --batch 1 --opset=12

```bash
atc --model=new_mbr_weight.onnx --framework=5 --output=MBR_4G_wsl 
    --soc_version=Ascend310B4 --input_shape='input:1,3,256,256' 
    --input_format=NCHW --precision_mode=force_fp32 
    --insert_op_conf=insert_op.cfg --fusion_switch_file=fusion_off.json

## 功能

- 模型加载和推理
- 图像预处理和后处理
- Ascend SDK 推理接口对接

## 使用方法

见 `main.cpp` 和 `README.md`。

## 依赖

- Ascend C++ SDK
- OpenCV
- CMake
