import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\Pycharm\Project\iFLYTEK\yolov11-1\yolo11n.pt')
    model.train(data=r'D:\Pycharm\Project\iFLYTEK\yolov11-1\dataset\dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )