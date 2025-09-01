#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from ultralytics.nn.tasks import parse_model
from ultralytics.utils import yaml_load

def test_yolov7():
    print("Loading YOLOv7 YAML...")
    d = yaml_load('ultralytics/cfg/models/v7-/yolov7.yaml')
    print("YAML loaded successfully")
    
    print("Parsing model...")
    try:
        model, save = parse_model(d, ch=3, verbose=True)  # 打开详细输出
        print(f"SUCCESS! Model parsed with {len(model)} layers")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_yolov7()