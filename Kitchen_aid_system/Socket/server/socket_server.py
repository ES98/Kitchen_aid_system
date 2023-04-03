import socket
import select

server_addr = '192.168.190.1', 8000

# Create a socket with port and host bindings
def setupServer():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket created")
    try:
        s.bind(server_addr)
    except socket.error as msg:
        print(msg)
    return s


# Establish connection with a client
def setupConnection(s):
    s.listen(5)     # Allows five connections at a time
    print("Waiting for client")
    conn, addr = s.accept()
    return conn


# Get input from user
def GET():
    reply = input("Reply: ")
    return reply


def sendFile(filename, conn):
    f = open(filename, 'rb')
    line = f.read(1024)

    print("Beginning File Transfer")
    while line:
        conn.send(line)
        line = f.read(1024)
    f.close()
    print("Transfer Complete")


# Loop that sends & receives data
def dataTransfer(conn, s, mode):
    while True:
        # Send a File over the network
        if mode == "SEND":
            filename = conn.recv(1024)
            filename = filename.decode(encoding='utf-8')
            filename.strip()
            print("Requested File: ", filename)
            sendFile(filename, conn)
            # conn.send(bytes("1", 'utf-8'))
            conn.send(bytes("DONE", 'utf-8'))
            break

        # Chat between client and server
        elif mode == "CHAT":
            # Receive Data
            print("Connected with: ", conn)
            data = conn.recv(1024)
            data = data.decode(encoding='utf-8')
            data.strip()
            print("Client: " + data)
            command = str(data)
            if command == "QUIT":
                print("Server disconnecting")
                s.close()
                break

            # Send reply
            reply = GET()
            conn.send(bytes(reply, 'utf-8'))

    # conn.close()

BUF_SIZE = 1024
FUNCTION_CONTROL = 0

sock = setupServer()
c_sock = setupConnection(sock)
print("Connecting Established")

while True:
    try:
        if FUNCTION_CONTROL == 1 :
            dataTransfer(c_sock, sock, "CHAT")
            FUNCTION_CONTROL = 0
        elif FUNCTION_CONTROL == 2 :
            # readBuf = c_sock.recv(BUF_SIZE)
            dataTransfer(c_sock, sock, "SEND")
            FUNCTION_CONTROL = 0
            c_sock.send(bytes("Transfer complete", 'utf-8'))

        readBuf = c_sock.recv(BUF_SIZE)
        print("Client :", readBuf.decode('utf-8'))

        if str(readBuf, 'utf-8') == 'chat':
            FUNCTION_CONTROL = 1
            print("Chat mode")

        elif str(readBuf, 'utf-8') == 'file':
            FUNCTION_CONTROL = 2
            print("file transfer mode")

        elif str(readBuf, 'utf-8') == 'reset':
            FUNCTION_CONTROL = 0
            c_sock.send(bytes("Do reset", 'utf-8'))
            print("Do reset")

        elif str(readBuf, 'utf-8') == 'exit' :
            c_sock.close()

        else :
            c_sock.send(bytes("No request arrived", 'utf-8'))


    except:
        break