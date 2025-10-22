from ultralytics import YOLO

def train_custom_model():
    model = YOLO('yolov8m.pt')

    results = model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        project='models',
        name='chess_black_detector_v1',
        batch=8
    )

    print("Training completed.")
    print("Best model saved at 'models/chess_black_detector_v1/weights/best.pt'")

if __name__ == '__main__':
    train_custom_model()