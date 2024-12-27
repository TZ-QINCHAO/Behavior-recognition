import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-fs/ultralytics-main/runs/train/exp167/weights/best.pt') # select your model.pt path
    model.predict(source='/root/autodl-fs/ultralytics-main/dataset/SHUJUJI/images/test',
                  imgsz=640,
                  project='/root/autodl-fs/ultralytics-main/runs/detect',
                  name='exp',
                  save=True,
                  #conf=0.2,
                  #iou=0.7,
                  # visualize=True # visualize model features maps
                )