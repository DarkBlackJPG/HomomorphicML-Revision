from timeit import timeit
from numpy import add, byte, copyto
from DataSender import DataSender
from KNN import *
import pycrypto
import pickle
from AES import AES
import pickle
import random
import time
from DataSender import DataSender
from DataReceiver import DataReceiver
from utility import *
from collections import Sequence

class PalisadeKNN(KNN):

    START_INT = 439 # Prime
    END_INT = 19319 # Prime

    def __init__(self, name: str, k: int = 2) -> None:
        super().__init__(name, k)
        self.cryptographic_params = None
        self.crypto_context = None
        self.password = 'XDK123XDK'
        self.disclosed_iv = b'\xc8\x8b\xf7\rn\xaf.\xd67\xb0\xd8\xd7\xd8\x17\x1cf'
    
    def encrypt_data(self, X) -> object:
        if isinstance(X, np.ndarray):
            raise ValueError

        if not isinstance(X, Sequence):
            return self.crypto_context.encryptFrac(X)

        encrypted_data = []
        for row in X:
            if isinstance(row, Sequence): # and len(row) > 1: 
                encrypted_data.append(self.crypto_context.Encrypt(row))
            else:
                print(row)
                print('Invalid data for encryption, must be list')
                raise ValueError
        return encrypted_data
    
    
    def encrypted_fit(self) -> None:
        print('Homomorphic fitting not supported - Will plaintext fit then encrypt weights')
        encrypted_fit_timer = Timer()
        encrypted_fit_timer.start()

        self.plaintext_fit()

        encrypted_fit_timer.finish()
        self.time_tracking[self.ENCFIT] = encrypted_fit_timer.get_time_in(Timer.TIMEFORMAT_MS)

    # Normalization must be done before this
    def encrypted_predict(self, X) -> object:
        self.general_timer.start()

        if len(X) > 1:
            print('Expected only one point. No batching! shape(1, <number_features>)')
            pass
        
        encrypted_X = X[0]

        is_encrypted = isinstance(encrypted_X[0], pycrypto.Ciphertext)
        if not is_encrypted:
            encrypted_X = self.encrypt_data(X)
        

        # Calculate distance
        encrypted_distance = []
        for my_data in self.encrypted_X:
            encrypted_distance.append(self.encrypted_distance(encrypted_X[0], my_data))

        confused_data = self.confuse_data(encrypted_distance, self.password)

        self.general_timer.finish()
        self.time_tracking[self.ENCPREDICT] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
        return confused_data
    
    def confuse_data(self, X, password: str):
        return_array = []
        for i in range(0, len(X)):
            return_array.append((X[i], self.encrypted_y[i]))
        
        return return_array

    def get_HE_context_bytes(self):
        context_dict = dict()

        context_dict['context'] =  self.crypto_context.to_bytes_context()
        context_dict['publicKey'] = self.crypto_context.to_bytes_publicKey()
        context_dict['relinKey'] = self.crypto_context.to_bytes_relinKey()
        # context_dict['rotateKey'] = self.crypto_context.to_bytes_rotateKey()
        context_dict['secretKey'] = self.crypto_context.to_bytes_secretKey()

        context_bytes = self.pickle_data(context_dict)

        return context_bytes

    def send_data_and_return_sorted(self, data, additional_data: dict = None):
        # data ==> (HE, AES) <-- Tuple
        data_packet = dict()

        server_operation = 'pyfhel/sort' 
        data_packet['context_data'] = self.get_HE_context_bytes()
        data_packet['necessary_data'] = data
        data_packet['additional_data'] = additional_data

        # Password & IV must be disclosed beforehand, either in person or via KDC
        ds = DataSender({'host': 'localhost', 'port': 8080})
        dr = DataReceiver({'host': 'localhost', 'port': 8081})
        password = self.password
        aes = AES()

        prepared_data = self.pickle_data(data_packet)
        encrypted_data = aes.encrypt(prepared_data, password, self.disclosed_iv)['data']
        prepared_encrypted = pickle.dumps((encrypted_data, server_operation))
        # SEND DATA
        ds.send_data(prepared_encrypted)

        # RECEIVE DATA
        resulting_vector = dr.connect()
        resulting_vector = pickle.loads(resulting_vector)
        resulting_data = aes.decrypt(resulting_vector, self.password, self.disclosed_iv)
        sorted_array = self.unpickle_data(resulting_data)

        return sorted_array

    def decrypt(self, X) -> object:
        self.general_timer.start()
        result = None
        if isinstance(X, list):
            print('Expected ciphertext object, got list')
            raise ValueError
        else:
            result = self.crypto_context.Decrypt(X)
        
        self.general_timer.finish()
        self.time_tracking[self.DECDATA] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS) 
        return result


    def encrypted_distance(self, X1, X2):
        temp = self.crypto_context.EvalSub(X1, X2)
        temp = self.crypto_context.EvalMultAndRelinearize(temp, temp)
        temp = self.crypto_context.EvalSum(temp, self.nextPowerOfTwo)

        return temp

    def symmetric_encrypt(self, X, password) -> object:
        if not isinstance(X, bytes):
            raise ValueError
        aes = AES()
        result = aes.encrypt(X, password)

        return result['data'], result['iv']

    def symmetric_decrypt(self, X, password, iv) -> object:
        if not isinstance(X, bytes):
            raise ValueError
        aes = AES()
        result = aes.decrypt(X, password, iv)

        return result
        
    def pickle_data(self, X) -> object:
        return pickle.dumps(X)
        
    def unpickle_data(self, X) -> object:
        return pickle.loads(X)

    def initialize(self, X, y, cryptographic_params: dict = None, data_normalization: dict = None) -> None:
        super().initialize(X, y, None, data_normalization)

        self.cryptographic_params = cryptographic_params
        if cryptographic_params is None:
            self.cryptographic_params = {}
            self.cryptographic_params['maxDepth'] = 1
            self.cryptographic_params['scaleFactor'] = 50
        
        self.general_timer.start()
        self.crypto_context = pycrypto.CKKSwrapper()
        self.crypto_context.KeyGen(self.cryptographic_params['maxDepth'], self.cryptographic_params['scaleFactor'], self.nextPowerOfTwo)
        self.general_timer.finish()
        self.time_tracking[self.KEYGEN] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.general_timer.start()
        self.encrypted_X = self.encrypt_data(np_to_list(self.X))
        self.general_timer.finish()
        self.time_tracking[self.DATAENCX] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.general_timer.start()
        prepared_y = np_to_list(self.y)
        prepared_y = [[element] for element in self.y] # must accept array, might change for fitting 
        self.encrypted_y = self.encrypt_data(prepared_y)
        self.general_timer.finish()
        self.time_tracking[self.DATAENCY] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
  
    
    def encrypted_test(k = 2):
        train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data('data/input.csv', 'data/check.csv', 'data/input.csv', 'data/check.csv')

        svm = PyfhelKNN('Pyfhel K nearest neighbors', k)
        svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'none', 'y': 'none'})
        svm.encrypted_fit()
        print(np.sign(svm.decrypt(svm.encrypted_predict([np_to_list(svm.X[23])]))), svm.y[23])

        print('------' + svm.algorithm_name + '--------')
        #print(f'TP: {TP}\nTF: {TF}\nFP: {FP}\nFF: {FF}\nNumber of elements: {TP + TF + FF + FP}\nCorrect predictions: {TP + TF}\nIncorrect prediction: {FF + FP}\nSuccess rate: {(TP + TF)/(TP + TF + FF + FP)}')
        svm.print_time_tracking_data()
        # print('-- Prediction times: [Includes latency for internal clock]')
        #pretty_table(plaintext_data_time, ['Value index', 'Time [MS]'])
import os
if __name__ == '__main__':
    cryptographic_params = {}
    cryptographic_params['maxDepth'] = 1
    cryptographic_params['scaleFactor'] = 50
    
    crypto_context = pycrypto.CKKSwrapper()
    crypto_context.KeyGen(cryptographic_params['maxDepth'], cryptographic_params['scaleFactor'], 64)

    encrypted_array = crypto_context.Encrypt([1,2,3,4])
    print(encrypted_array)
    decrypted = crypto_context.Decrypt(encrypted_array)
    print(decrypted[0:4])
    
    neu = pickle.dumps(crypto_context)
    neuDecrypr = neu.Decrypt(encrypted_array)
    print(neuDecrypr[0:4])

