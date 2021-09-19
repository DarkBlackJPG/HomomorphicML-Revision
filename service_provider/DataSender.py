import pickle
import socket
import sys
import time
class DataSender:
    def __init__(self, connection_details: dict = None) -> None:
        self.TRANSFER_SIZE = 20971520 # 20MB
        if connection_details is None:
            connection_details = {}
            connection_details['host'] = socket.gethostname()
            connection_details['port'] = 8080
        
        self.connection_details = connection_details

        


    def send_data(self, data):
        self.connection_socket = socket.socket()
        try:
            self.connection_socket.connect((self.connection_details['host'], int(self.connection_details['port'])))
            status = self.connection_socket.sendall(data)
            return True if status is None else False
        except Exception as e:
            print(e)
            return False
        finally:
            self.connection_socket.close()
        