# 可用单次

# import socket
# import os

# # def receive_image(server_socket, filename):
# #     with open(filename, 'wb') as f:
# #         while True:
# #             data = server_socket.recv(1024)
# #             if not data:
# #                 break
# #             f.write(data)

# def receive_image(server_socket, relative_path, filename):
#     if not os.path.exists(relative_path):
#         os.makedirs(relative_path)  # 创建文件夹
#     full_path = os.path.join(relative_path, filename)  # 构建完整路径

#     with open(full_path, 'wb') as f:
#         while True:
#             data = server_socket.recv(1024)
#             if not data:
#                 break
#             f.write(data)


# def main():
#     host = ''  # 服务器IP地址
#     port = 8080  # 服务器端口号

#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.bind((host, port))
#     server_socket.listen(1)

#     print('等待客户端连接...')
#     client_socket, addr = server_socket.accept()
#     print('与客户端建立连接:', addr)

#     # 接收图片
#     receive_image(client_socket,'DATA', 'received_image.jpg')

#     print('图片接收完成')

#     client_socket.close()
#     server_socket.close()

# if __name__ == '__main__':
#     main()


# 加入循环
import socket
import os


def receive_image(server_socket, relative_path, filename):
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)  # 创建文件夹
    full_path = os.path.join(relative_path, filename)  # 构建完整路径

    with open(full_path, 'wb') as f:
        while True:
            data = server_socket.recv(1024)
            if not data:
                break
            f.write(data)


def main():
    host = ''  # 服务器IP地址
    port = 8080  # 服务器端口号
    received_image_directory = 'DATA'  # 接收到的图片存放目录

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print('等待客户端连接...')

    img_count = 0

    while True:
        client_socket, addr = server_socket.accept()
        print('与客户端建立连接:', addr)

        # 接收图片
        # receive_image(client_socket, received_image_directory, f'received_image_{addr[0]}_{addr[1]}.jpg')
        img_count += 1
        receive_image(client_socket, received_image_directory, f'{img_count}.jpg')
        print('图片接收完成')

        client_socket.close()


if __name__ == '__main__':
    main()


###############################################################################

# import socket
# import os

# def receive_image(server_socket, relative_path):
#     if not os.path.exists(relative_path):
#         os.makedirs(relative_path)  # 创建文件夹

#     filename = 'received_image.jpg'
#     full_path = os.path.join(relative_path, filename)  # 构建完整路径

#     with open(full_path, 'wb') as f:
#         while True:
#             data = server_socket.recv(1024)
#             if not data:
#                 break
#             f.write(data)

# def main():
#     host = ''  # 服务器IP地址
#     port = 8080  # 服务器端口号
#     received_image_directory = 'received_images'  # 接收到的图片存放目录

#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.bind((host, port))
#     server_socket.listen(1)

#     print('等待客户端连接...')
#     client_socket, addr = server_socket.accept()
#     print('与客户端建立连接:', addr)

#     # 接收图片
#     receive_image(client_socket, received_image_directory)

#     print('图片接收完成')

#     client_socket.close()
#     server_socket.close()

# if __name__ == '__main__':
#     main()
