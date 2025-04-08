from ultralytics import YOLO
import cv2
#model = YOLO(r'D:\Pycharm\Project\iFLYTEK\yolov11-1\yolo11x.pt')
model = YOLO(r'D:\Pycharm\Project\iFLYTEK\yolov11-1\runs\train\train5\weights\best.pt')
cap=cv2.VideoCapture(0)
while True:
    # 读取一帧
    success, frame = cap.read()
    if not success:
        break

    # 对当前帧进行目标检测
    results = model.predict(source=frame)

    # 获取绘制好结果的图像帧
    annotated_frame = results[0].plot()

    # 显示结果
    cv2.imshow('Faces', annotated_frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()