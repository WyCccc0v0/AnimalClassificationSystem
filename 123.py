from ultralytics import YOLO

# 加载官方预训练模型
model = YOLO(r"D:\Pycharm\Project\iFLYTEK\yolov11-1\yolov11s_cbam.pt").model

# 打印模型结构，确认原始结构
print(model)
