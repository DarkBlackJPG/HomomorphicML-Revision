import pickle
from DataReceiver import DataReceiver
from DataSender import DataSender
from ServerAES import AES
from Pyfhel import Pyfhel, PyCtxt, PyPtxt

class ServiceServer:
    password = 'XDK123XDK'
    iv = b'\xc8\x8b\xf7\rn\xaf.\xd67\xb0\xd8\xd7\xd8\x17\x1cf'
    def __init__(self) -> None:
        pass

    def restore_pyfhel_context(self, context):
        self.Pyfhel = Pyfhel()
        context = pickle.loads(context)
        self.Pyfhel.from_bytes_context(context['context'])
        self.Pyfhel.from_bytes_publicKey(context['publicKey'])
        self.Pyfhel.from_bytes_relinKey(context['relinKey'])
        # self.Pyfhel.from_bytes_rotateKey(context['rotateKey'])
        self.Pyfhel.from_bytes_secretKey(context['secretKey'])
    
    def run(self):
        dr = DataReceiver({'host': 'localhost', 'port': 8080})
        ds = DataSender({'host': 'localhost', 'port': 8081})
        print('DEBUG: Server running...')
        while True:
            print('Waiting connection')
            client_data = dr.connect()
            client_data = pickle.loads(client_data)
            operation = None
            context_data = None
            necessary_data = None
            additional_data = None
            print('Enter all')
            if client_data[1] == 'pyfhel/sort':
                print(client_data[1])
                client_data = client_data[0]
                password = ServiceServer.password
                iv = ServiceServer.iv

                aes = AES()

                pickled_received_data = aes.decrypt(client_data, ServiceServer.password, ServiceServer.iv)
                unpickled_data = pickle.loads(pickled_received_data)

                context_data = unpickled_data['context_data']
                necessary_data = unpickled_data['necessary_data']
                additional_data = unpickled_data['additional_data']

                self.restore_pyfhel_context(context_data)
                
                # Necessary Data tuple array (HE(<Data to sort>), AES(<Class>))
                decrypted_array = [(self.Pyfhel.decryptFrac(element[0]), element[1]) for element in necessary_data]
                sorted_data = sorted(decrypted_array, key=lambda tup: tup[0])
                sorted_data = [element[1] for element in sorted_data]
                pickled_sorted_data = pickle.dumps(sorted_data)
                encrypted_data = aes.encrypt(pickled_sorted_data, ServiceServer.password, ServiceServer.iv)['data']
                encrypted_data = pickle.dumps(encrypted_data)
                ds.send_data(encrypted_data)

                print('Operation Done')


if __name__ == '__main__':
    serviceProvider = ServiceServer()
    serviceProvider.run()
