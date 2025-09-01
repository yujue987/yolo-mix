from ultralytics import YOLO

# Load a model
# build a new model from YAML
model = YOLO("yolov1.yaml")

# load a pretrained model (recommended for training)
# model = YOLO("yolo11n.pt")

# build from YAML and transfer weights
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")

# Train the model
results = model.train(data="voc.yaml", epochs=10, imgsz=640)
