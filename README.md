# YOLOV5_Ascend_CPP

1. 模型转换：

// pt-->onnx
python export.py --include onnx --weights yolov5s.pt --img 640 --batch 1 --opset=12

// onnx-->om
atc --model=new_mbr_weight.onnx --framework=5 --output=MBR_4G_wsl 
    --soc_version=Ascend310B4 --input_shape='input:1,3,256,256' 
    --input_format=NCHW --precision_mode=force_fp32 
    --insert_op_conf=insert_op.cfg --fusion_switch_file=fusion_off.json