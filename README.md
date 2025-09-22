# yolo-mix

## 大致介绍

目前实现了v1,v2,v4,v7在配置文件中有，文件夹在相应的后面有“-”
是在ultralytics的基础上做的，使用方法是一样的。
修改的内容在`修改`文件中有

## 使用

训练相关参数在**`ultralytics/cfg/default.yaml`**中，可以在https://docs.ultralytics.com/zh/modes/train/#train-settings 找相关参数的介绍
```python
#最基础的使用是
from ultralytics import YOLO

# Load a model
model = YOLO("path/to/last.pt")  # load a partially trained model

# Resume training
results = model.train(resume=True)
#参考 https://docs.ultralytics.com/zh/
```

## 对于 `模块介绍` 文件夹的介绍

主要是介绍其中的模块
对于 `nn/modle/` 下的每一个文件，在 `模块介绍` 中都有同名文件与其对应，其中是对于文件中模块的介绍。
