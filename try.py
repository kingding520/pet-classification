import mindspore
import mindspore.nn as nn

# 尝试加载
try:
    net = mindspore.load_checkpoint("mobilenetv2.mindir")
    print("模型加载成功")
except Exception as e:
    print(f"加载失败: {e}")