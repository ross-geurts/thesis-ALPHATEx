from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(
    data="dataset_split/data.yaml", 
    epochs=100, 
    imgsz=640,
    cfg="parameters.yaml",
    save_json=True
)