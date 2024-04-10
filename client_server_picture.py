

import socket
import os

def send_single_image(client_socket, directory):
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    if len(image_files) != 1:
        print("There should be exactly one image in the directory.")
        return
    
    image_filename = image_files[0]
    image_path = os.path.join(directory, image_filename)

    with open(image_path, 'rb') as f:
        image_data = f.read()

    client_socket.sendall(image_data)


def delete_files_in_directory(directory_path):
    try:
        # 遍历目录中的所有文件
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # 如果是文件则删除
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        
        #print(f"All files in {directory_path} have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")



def main():
    host = '110.41.9.233'  # 服务器IP地址
    port = 8080  # 服务器端口号
    image_directory = 'images'  # 图片所在目录

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # 发送图片
    send_single_image(client_socket, image_directory)

    print('图片发送完成')

    client_socket.close()
    
    # 删除已经发送文件
    # 获取当前脚本所在目录
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(script_directory, image_directory)
    delete_files_in_directory(data_folder_path)

if __name__ == '__main__':
    main()






















# import socket
# import os

# def send_image(client_socket, filename):
#     with open(filename, 'rb') as f:
#         while True:
#             data = f.read(1024)
#             if not data:
#                 break
#             client_socket.sendall(data)

# def main():
#     host = '110.41.9.233'  # 服务器IP地址
#     port = 8080  # 服务器端口号
#     image_directory = 'images'  # 图片所在目录
#     image_filename = '1.jpg'  # 图片文件名

#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client_socket.connect((host, port))

#     # 发送图片文件名
#     client_socket.sendall(image_filename.encode())

#     # 发送图片
#     send_image(client_socket, os.path.join(image_directory, image_filename))

#     print('图片发送完成')

#     client_socket.close()

# if __name__ == '__main__':
#     main()
