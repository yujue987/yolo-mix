from ultralytics import YOLO

# 基础使用方式
# Load a model
# build a new model from YAML
# model = YOLO("yolov11n.yaml")

# load a pretrained model (recommended for training)
# model = YOLO("yolo11n.pt")

# build from YAML and transfer weights
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")

# Train the model
# results = model.train(data="coco8.yaml", epochs=1, imgsz=640)

# #yolov1
# model = YOLO("yolov1.yaml")
# results = model.train(data="VOC.yaml", epochs=1, imgsz=640)

# #yolov2
# model = YOLO("yolov2.yaml")
# results = model.train(data="coco8.yaml", epochs=1, imgsz=640)

# #yolov4
# model = YOLO("yolov4.yaml")
# results = model.train(data="coco8.yaml", epochs=1, imgsz=640)

#yolov7
model = YOLO("ultralytics/cfg/models/v7-/yolov7.yaml")
results = model.train(data="coco8.yaml", epochs=1, imgsz=640, batch=1, device='cpu')