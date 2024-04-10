# import socket
# import os

# sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sk.bind(('', 8080))  # 地址
# sk.listen(5)

# conn, addr = sk.accept()

# while True:
#     file_data = conn.recv(10240000)

#     if file_data == b'bye':
#         conn.send(b'bye')
#         break

#     # filename = input("Enter a filename to save: ")

#     # 输入要传输过来要保存的文件名即可(无需完整地址): testdata.txt

#     filename = os.path.join(
#         'C:\\Users\\Legion\\Desktop\\server', input("Enter a filename to save: "))
#     # server_directory = 'files'

#     # filename = os.path.join(
#     #     server_directory, input("Enter a filename to save: "))
#     with open(filename, 'wb') as file:
#         file.write(file_data)
#         conn.send(b'File received and saved.')

# conn.close()
# sk.close()





filepath = 'C:\Users\Legion\Desktop\ppp.png'