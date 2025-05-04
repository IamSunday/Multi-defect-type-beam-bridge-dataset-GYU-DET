from ultralytics import YOLO
if __name__ == '__main__':
    # Use the model
    model = YOLO("yolo11.yaml")
    results = model.train(data='./GZ-DET.yaml',epochs=300)