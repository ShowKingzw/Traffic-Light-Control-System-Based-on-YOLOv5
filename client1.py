import cv2
import socket
import struct

client_socket = socket.socket()
client_socket.connect(('127.0.0.1', 8002))

connection = client_socket.makefile('wb')

try:
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换为JPEG格式
        # img_str = cv2.imencode('.jpg', frame)[1].tostring()
        img_str = cv2.imencode('.jpg', frame)[1].tobytes()
        # 获得图像长度
        s = struct.pack('<L', len(img_str))
        
        # 将图像长度传输到服务器
        connection.write(s)
        connection.flush()
        # 传输图像数据
        connection.write(img_str)
        connection.flush()

except Exception as e:
    print(e)
finally:
    connection.close()
    client_socket.close()
