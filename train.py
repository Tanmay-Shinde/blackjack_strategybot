from ultralytics import YOLO
# import onnx

def train():
    # load model:
    model = YOLO("yolov8n.yaml")
    # model.export(format="onnx")

    # use config file:
    data_yaml_file = "C:/Users/tanma/PycharmProjects/blackjack_strategybot/playing_cards_data/data.yaml"

    project = "C:/Users/tanma/PycharmProjects/blackjack_strategybot/playing_cards_data"
    experiment = "card_detection_model"

    batch_size = 32

    results = model.train(
        data=data_yaml_file,
        epochs=50,
        project=project,
        name=experiment,
        batch=batch_size,
        patience=5,
        imgsz=640,
        verbose=True,
        val=True
    )

if __name__ == "__main__":
    train()