import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO('/root/autodl-fs/ultralytics-main/ultralytics/cfg/models/v8/yolov8n-bifpn-SHUJUJI.yaml')
    model = YOLO('/root/autodl-fs/ultralytics-main/ultralytics/cfg/models/v8/yolov8n-C2f-EIEM-SHUJUJI.yaml')

    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/root/autodl-fs/ultralytics-main/dataset/XINGWEI.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                patience=50,
                # patience=0, # close earlystop
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='/root/autodl-fs/ultralytics-main/runs/train',
                name='exp',
                )