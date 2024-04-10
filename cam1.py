import cv2

camera = cv2.VideoCapture(0)  # 0表示默认摄像头

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # 在这里进行图像处理或显示
    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
