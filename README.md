# 宠物分类项目 (Pet Classification)

基于MindSpore的猫狗图像分类项目，支持模型训练和MindSpore Lite推理部署。

## 项目简介

本项目使用MobileNetV2网络对猫狗图像进行分类，支持从数据预处理、模型训练到模型转换的完整流程。

## environment requirements
```bash
conda create -n pet_classification python=3.9
conda activate pet_classification
```
1.python 3.9    
2.[mindspore 2.9.0](https://www.mindspore.cn/versions#2.9.0) [lite](https://www.mindspore.cn/lite)    
3.matplotlib    
4.imageio    
5.opencv-python    
6.easydict

## structure
```bash
pet-classification/
├── code/                       # 代码目录
│   ├── preprocessing_dataset.py   # 数据清洗脚本
│   ├── train.py                   # 训练脚本
│   └── mobilenetv2.mindir        # 导出的模型文件
├── data/                         # 数据目录
├── models/                       # 保存的模型
├── scripts/                      # 工具脚本
│   └── convert_model.bat        # 模型转换脚本
├── requirements.txt              # Python依赖
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件
```
## lab
1.清洗脚本
```bash
python preprocessing_dataset.py D:\MindSporePetClassification\kagglecatsanddogs_3367a.zip
```
2.训练模型
```bash
python train.py
```
3.模型转换
```bash
call "D:\MindSporePetClassification\mindspore-lite-2.9.0-win-x64\tools\converter\converter_lite.exe" --fmk=MINDIR --modelFile="D:\MindSporePetClassification\code\mobilenetv2.mindir" --outputFile=pet
```
## trouble
工具链不统一
最开始使用最新的mindspore 2.9.0成功生成mindir文件。但由于mindspore-lite的converter版本落后而无法转换模型。
通过下载最新的工具以转换。
其中遇到dll缺失问题。通过混用新旧converter依赖库临时解决。得到pet文件