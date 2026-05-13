# pet-classification

## requirements
1.python 3.9    
2.[mindspore 2.9.0](https://www.mindspore.cn/versions#2.9.0) [lite](https://www.mindspore.cn/lite)    
3.matplotlib    
4.imageio    
5.opencv-python    
6.easydict


## trouble
工具链不统一
最开始使用最新的mindspore 2.9.0成功生成mindir文件。但由于mindspore-lite的converter版本落后而无法转换模型。
通过下载最新的工具以转换。
其中遇到dll缺失问题。通过混用新旧converter依赖库临时解决。得到pet文件