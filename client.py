# import socket
# import os
# sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sk.connect(('110.41.9.233',8080)) # server 地址

# while True:
#     user_input = input('>>>') # 输入的是文件地址: C:\test\testdata.txt
#     if user_input == 'bye':
#         sk.send(b'bye')
#         break

#     try:
#         # user_input = r'' + user_input
#         with open(user_input, 'rb') as file:
#             file_data = file.read()
#             sk.send(file_data)
#     except FileNotFoundError:
#         print("File not found.")

#     response = sk.recv(1024).decode('utf-8')
#     print(response)

# sk.close()



import socket
import os

sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sk.connect(('110.41.9.233', 8080))  # server 地址

while True:
    user_input = input('>>>')  # 输入的是文件地址: C:\test\testdata.txt
    if user_input == 'bye':
        sk.send(b'bye')
        break

    try:
        with open(user_input, 'rb') as file:
            file_data = file.read()
            sk.send(file_data)
    except FileNotFoundError:
        print("File not found.")

    response = sk.recv(1024)  # 接收二进制数据
    print(response.decode('utf-8'))  # 如果服务器返回的是文本信息，可以在这里进行解码显示

sk.close()
