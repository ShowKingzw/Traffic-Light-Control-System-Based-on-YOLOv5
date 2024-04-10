import socket

try:

    # open image
    myfile = open("C:\\Users\\Legion\\Desktop\\ppp.png", 'rb')
    bytes = myfile.read()
    size = len(bytes)

    # send image size to server
    socket.sendall("SIZE %s" % size)
    answer = socket.recv(4096)

    print('answer = %s' % answer)

    # send image to server
    if answer == 'GOT SIZE':
        socket.sendall(bytes)

        # check what server send
        answer = socket.recv(4096)
        print('answer = %s' % answer)

        if answer == 'GOT IMAGE':
            socket.sendall("BYE BYE ")
            print('Image successfully send to server')

    myfile.close()

finally:
    socket.close()
