import pickle

from numpy import byte
from DataReceiver import DataReceiver
from DataSender import DataSender
from ServerAES import AES
from Pyfhel import Pyfhel, PyCtxt, PyPtxt
import seal
import os

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

    def save_bytes_to_file(self, filename, bytes):
        f = open(filename, 'wb')
        f.write(bytes)
        f.close()
        pass

    def remove_file(self, filename):
        os.remove(filename)
        pass

    def restore_seal_context(self, context):
        self.scale = context['scale']
        self.slot_count = context['slot_count']
        public_key_bytes = context['public_key']  # bytes
        params_bytes = context['params']          # bytes
        secret_key_bytes = context['secret_key']  # bytes
        relin_keys_bytes = context['relin_keys']  # bytes

        self.save_bytes_to_file('public_key', public_key_bytes)
        self.save_bytes_to_file('params', params_bytes)
        self.save_bytes_to_file('secret_key', secret_key_bytes)
        self.save_bytes_to_file('relin_keys', relin_keys_bytes)

        self.public_key = seal.PublicKey()
        self.secret_key = seal.SecretKey()
        self.relin_key = seal.RelinKeys()
        self.encryption_parameters = seal.EncryptionParameters(seal.scheme_type.ckks)
        self.encryption_parameters.load('params')
        
        self.context = seal.SEALContext(self.encryption_parameters)        
        self.ckks_encoder = seal.CKKSEncoder(self.context)


        self.public_key.load(self.context, 'public_key')
        self.secret_key.load(self.context, 'secret_key')
        self.relin_key.load(self.context, 'relin_keys')

        self.encryptor = seal.Encryptor(self.context, self.public_key)
        self.evaluator = seal.Evaluator(self.context)
        self.decryptor = seal.Decryptor(self.context, self.relin_key)
        
        self.remove_file('public_key')
        self.remove_file('params')
        self.remove_file('secret_key')
        self.remove_file('relin_keys')

    def seal_decrypt_tuple(self, necessary_data):
        # Necessary Data tuple array (HE(<Data to sort>), AES(<Class>))
        decrypted_data = []
        for instance in necessary_data:
            he_data = instance[0]
            aes_data = instance[1]

            decrypted = self.decryptor.decryptor(he_data)
            decoded = self.ckks_encoder.decode(decrypted)

            # Need to get [0] because of size of data
            decrypted.append((decoded[0], aes_data))
        return decrypted
    
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

                aes = AES()

                pickled_received_data = aes.decrypt(client_data, ServiceServer.password, ServiceServer.iv)
                unpickled_data = pickle.loads(pickled_received_data)

                context_data = unpickled_data['context_data']
                necessary_data = unpickled_data['necessary_data']
                additional_data = unpickled_data['additional_data'] # not used

                self.restore_pyfhel_context(context_data)
                
                # Necessary Data tuple array (HE(<Data to sort>), AES(<Class>))
                decrypted_array = [(self.Pyfhel.decryptFrac(element[0]), element[1]) for element in necessary_data]
                sorted_data = sorted(decrypted_array, key=lambda tup: tup[0])
                sorted_data = [element[1] for element in sorted_data]
                pickled_sorted_data = pickle.dumps(sorted_data)
                encrypted_data = aes.encrypt(pickled_sorted_data, ServiceServer.password, ServiceServer.iv)['data']
                encrypted_data = pickle.dumps(encrypted_data)
                ds.send_data(encrypted_data)
                self.Pyfhel = None
                print('Operation Done')


            elif client_data[1] == 'seal/sort':
                print(client_data[1])
                client_data = client_data[0]

                aes = AES()

                pickled_received_data = aes.decrypt(client_data, ServiceServer.password, ServiceServer.iv)
                unpickled_data = pickle.loads(pickled_received_data)

                context_data = unpickled_data['context_data']
                necessary_data = unpickled_data['necessary_data']
                additional_data = unpickled_data['additional_data'] # not used

                self.restore_seal_context(context_data)
                
                # Necessary Data tuple array (HE(<Data to sort>), AES(<Class>))
                # decrypted_array = [(self.Pyfhel.decryptFrac(element[0]), element[1]) for element in necessary_data]
                decrypted_array = self.seal_decrypt_tuple(necessary_data)
                sorted_data = sorted(decrypted_array, key=lambda tup: tup[0])
                sorted_data = [element[1] for element in sorted_data]
                pickled_sorted_data = pickle.dumps(sorted_data)
                encrypted_data = aes.encrypt(pickled_sorted_data, ServiceServer.password, ServiceServer.iv)['data']
                encrypted_data = pickle.dumps(encrypted_data)
                ds.send_data(encrypted_data)


if __name__ == '__main__':
    serviceProvider = ServiceServer()
    serviceProvider.run()
