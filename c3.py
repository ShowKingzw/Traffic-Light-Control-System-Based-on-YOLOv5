import socket
import struct
import sys
import os
import time
 
 
if __name__ == '__main__':
 
    file_name = sys.argv[1]
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('110.41.9.233', 8080))
 
    # 定义文件头信息；
    file_head = struct.pack('128sl', os.path.basename(file_name).encode(),
                        os.stat(file_name).st_size)
 
    sock.send(file_head)
 
    read_file = open(file_name, "rb")
 
    while True:
        # time.sleep(1)
        file_data = read_file.read(10240)
 
        if not file_data:
            break
 
        sock.send(file_data)
 
    read_file.close()
    sock.close()