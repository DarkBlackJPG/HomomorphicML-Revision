import socket
import sys
import threading

class DataReceiver:
    def __init__(self, connection_details: dict = None) -> None:
        self.TRANSFER_SIZE = 20971520 # 20MB
        if connection_details is None:
            connection_details = {}
            connection_details['host'] = socket.gethostname()
            connection_details['port'] = 8080
        
        self.connection_details = connection_details

    def receive_data(self, client_socket: socket.socket):
        data_list = b''
        data = client_socket.recv(self.TRANSFER_SIZE)
        data_list += data
        while len(data) > 0:
            data = client_socket.recv(self.TRANSFER_SIZE)
            data_list += data
        return data_list

    def connect(self):
        self.socket_connection = socket.socket()
        self.socket_connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket_connection.bind((self.connection_details['host'], self.connection_details['port']))
        self.socket_connection.listen(1)
        client_socket, _ = self.socket_connection.accept()
        data = self.receive_data(client_socket)
        self.socket_connection.close()
        return data

        

        

        
