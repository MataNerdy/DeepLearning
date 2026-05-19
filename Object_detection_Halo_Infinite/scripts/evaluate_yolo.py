import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/halo_yolo/config.yaml")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
