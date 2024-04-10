import socket
import struct
import io
import logging
import socketserver
from threading import Condition
from http import server

# 创建服务器套接字
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8002))
server_socket.listen(0)

connection = server_socket.accept()[0].makefile('rb')

PAGE = """\
<html>
<head>
<title>Camera MJPEG Streaming Demo</title>
</head>
<body>
<h1>Camera MJPEG Streaming Demo</h1>
<img src="img_str.mjpg" width="640" height="480" />
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            
            # Corrected Content-Type setting
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            
            self.end_headers()
            try:
                while True:
                    # 获取图像长度
                    image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
                    if not image_len:
                        break

                    image_stream = io.BytesIO()
                    # 读取图像数据
                    image_stream.write(connection.read(image_len))

                    image_stream.seek(0)
                    img_data = image_stream.getvalue()
                    # 写入HTTP响应
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(img_data))
                    self.end_headers()
                    self.wfile.write(img_data)
                    self.wfile.write(b'\r\n')

            except Exception as e:
                logging.warning('Error streaming client %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

# class StreamingHandler(server.BaseHTTPRequestHandler):
#     def do_GET(self):
#         if self.path == '/':
#             self.send_response(301)
#             self.send_header('Location', '/web1.html')
#             self.end_headers()
#         elif self.path == '/index.html':
#             content = PAGE.encode('utf-8')
#             self.send_response(200)
#             self.send_header('Content-Type', 'text/html')
#             self.send_header('Content-Length', len(content))
#             self.end_headers()
#             self.wfile.write(content)
#         elif self.path == '/stream.mjpg':
#             self.send_response(200)
#             self.send_header('Age', 0)
#             self.send_header('Cache-Control', 'no-cache, private')
#             self.send_header('Pragma', 'no-cache')
#             self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
#             self.end_headers()
#             try:
#                 while True:
#                     # 获取图像长度
#                     image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
#                     if not image_len:
#                         break

#                     image_stream = io.BytesIO()
#                     # 读取图像数据
#                     image_stream.write(connection.read(image_len))

#                     image_stream.seek(0)
#                     img_data = image_stream.getvalue()
#                     # 写入HTTP响应
#                     self.wfile.write(b'--FRAME\r\n')
#                     self.send_header('Content-Type', 'image/jpeg')
#                     self.send_header('Content-Length', len(img_data))
#                     self.end_headers()
#                     self.wfile.write(img_data)
#                     self.wfile.write(b'\r\n')

#             except Exception as e:
#                 logging.warning('Error streaming client %s: %s', self.client_address, str(e))
#         else:
#             self.send_error(404)
#             self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

output = StreamingOutput()

try:
    address = ('0.0.0.0', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
except Exception as e:
    print(e)
