from timeit import timeit
from numpy import add, byte
from DataSender import DataSender
from KNN import *
from Pyfhel import *
import pickle
from AES import AES
import pickle
import random
import time
from DataSender import DataSender
from DataReceiver import DataReceiver
from Crypto import Random


class PyfhelKNN(KNN):

    START_INT = 439 # Prime
    END_INT = 19319 # Prime

    def __init__(self, name: str, k: int = 2) -> None:
        super().__init__(name, k)
        self.cryptographic_params = None
        self.crypto_context = None
        self.password = 'XDK123XDK'
        self.disclosed_iv = b'\xc8\x8b\xf7\rn\xaf.\xd67\xb0\xd8\xd7\xd8\x17\x1cf'
    
    def encrypt_data(self, X) -> object:
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            return self.crypto_context.encryptFrac(X)

        encrypted_data = []
        for row in X:
            if isinstance(row, list) and len(row) > 1: 
                temp_array = []
                for cell in row:
                    temp_array.append(self.crypto_context.encryptFrac(cell))
                encrypted_data.append(temp_array)
            else:
                encrypted_data.append(self.crypto_context.encryptFrac(row))
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

        is_encrypted = isinstance(encrypted_X[0], PyCtxt)
        if not is_encrypted:
            encrypted_X = self.encrypt_data(encrypted_X)
        

        if len(encrypted_X) != self.n_features:
            print(f'Incorrect size of data. Expected {self.n_features} features')
            raise ValueError

        aes = AES()

        # Calculate distance
        encrypted_distance = []
        for my_data in self.encrypted_X:
            encrypted_distance.append(self.encrypted_distance(encrypted_X, my_data))

        confused_data = self.confuse_data(encrypted_distance, self.password)
        sorted_array = self.send_data_and_return_sorted(confused_data)
        sorted_array = sorted_array[: self.k]

        results = []
        for element in sorted_array:
            toDecrypt = element
            pickled = aes.decrypt(toDecrypt, self.password, self.disclosed_iv)
            results.append(pickle.loads(pickled))


        self.general_timer.finish()
        self.time_tracking[self.ENCPREDICT] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
        return results
    
    def confuse_data(self, X, password: str):
        random.seed(int(time.time()))
        self.confusion_random_value = random.random()
        self.permutation = [i for i in range(0, len(X))]
        random.shuffle(self.permutation)
        aes = AES()

        return_array = []
        for i in range(0, len(X)):
            temporary_confused_data = self.crypto_context.add(PyCtxt(X[i]), self.crypto_context.encryptFrac(self.confusion_random_value), True)
            # temporary_confused_data = self.crypto_context.encryptFrac(self.crypto_context.decryptFrac(temporary_confused_data))
            
            encrypted_class_bytes = pickle.dumps(self.encrypted_y[i])
            
            encryption_result = aes.encrypt(encrypted_class_bytes, password, self.disclosed_iv)

            # if self.confusion_encryption_details['iv'] is None:
            #     self.confusion_encryption_details['iv'] = encryption_result['iv']

            return_array.append((temporary_confused_data, encryption_result['data']))
        
        permutation = [None] * len(return_array)
        for i in range(len(X)):
            permutation[i] = return_array[self.permutation[i]]
        
        return permutation

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
            result = [self.crypto_context.decryptFrac(element) for element in X]
        else:
            result = self.crypto_context.decryptFrac(X)
        
        self.general_timer.finish()
        self.time_tracking[self.DECDATA] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS) 
        return result

    def initialize(self, X, y, cryptographic_params: dict = None, data_normalization: dict = None) -> None:
        super().initialize(X, y, None, data_normalization)

        self.cryptographic_params = cryptographic_params
        if cryptographic_params is None:
            self.cryptographic_params = {}
            self.cryptographic_params['p'] = 63
            self.cryptographic_params['m'] = 2048
            self.cryptographic_params['base'] = 2
            self.cryptographic_params['intDigits'] = 64
            self.cryptographic_params['fracDigits'] = 64
            self.cryptographic_params['relinKeySize'] = 6
            self.cryptographic_params['bitCount'] = 16
        
        self.general_timer.start()
        self.crypto_context = Pyfhel()
        self.crypto_context.contextGen(
            p=self.cryptographic_params['p'],
            m=self.cryptographic_params['m'],
            base=self.cryptographic_params['base'],
            intDigits=self.cryptographic_params['intDigits'],
            fracDigits=self.cryptographic_params['fracDigits'])
        self.crypto_context.keyGen()
        self.crypto_context.relinKeyGen(self.cryptographic_params['bitCount'], self.cryptographic_params['relinKeySize'])

        self.general_timer.finish()
        self.time_tracking[self.KEYGEN] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.general_timer.start()
        self.encrypted_X = self.encrypt_data(np_to_list(self.X))
        self.general_timer.finish()
        self.time_tracking[self.DATAENCX] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.general_timer.start()
        self.encrypted_y = self.encrypt_data(np_to_list(self.y))
        self.general_timer.finish()
        self.time_tracking[self.DATAENCY] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
  
    def encrypted_distance(self, X1, X2):
        X_1 = [PyCtxt(x) for x in X1] # Copy
        X_2 = [PyCtxt(x) for x in X2] # Copy
        encrypted_sum = None
        for x1_element, x2_element in zip(X_1, X_2):
            if not encrypted_sum:
                encrypted_sum = self.crypto_context.sub(x1_element, x2_element, True)
                encrypted_sum = self.crypto_context.square(encrypted_sum)
                encrypted_sum = ~ encrypted_sum
            else:
                temp_subtraction = self.crypto_context.sub(x1_element, x2_element, True)
                square = self.crypto_context.square(temp_subtraction, True)
                square = ~square
                encrypted_sum = self.crypto_context.encryptFrac(self.crypto_context.decryptFrac(encrypted_sum))
                encrypted_sum = self.crypto_context.add(square, encrypted_sum, True)
            encrypted_sum = self.crypto_context.encryptFrac(self.crypto_context.decryptFrac(encrypted_sum))
        return encrypted_sum

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
    
    def encrypted_test(k = 2):
        train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data('data/input.csv', 'data/check.csv', 'data/input.csv', 'data/check.csv')

        svm = PyfhelKNN('Pyfhel K nearest neighbors', k)
        svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'none', 'y': 'none'})
        svm.encrypted_fit()
        print(np.sign(svm.decrypt(svm.encrypted_predict([np_to_list(svm.X[23])]))), svm.y[23])
        #plaintext_data_time = []

        # TP = 0
        # TF = 0
        # FP = 0
        # FF = 0
        # timer = Timer()
        # it = 0
        # for point, expected in zip(test_input_data, test_train_check_data_file):
        #     timer.start()
        #     prediction = svm.plaintext_predict(point)
        #     timer.finish()
        #     plaintext_data_time.append([it, timer.get_time_in(Timer.TIMEFORMAT_MS)])
        #     if prediction == expected:
        #         if prediction > 0:
        #             TP += 1
        #         else:
        #             TF += 1
        #     else: 
        #         if prediction > 0:
        #             FP += 1
        #         else:
        #             FF += 1

        #     it += 1

        print('------' + svm.algorithm_name + '--------')
        #print(f'TP: {TP}\nTF: {TF}\nFP: {FP}\nFF: {FF}\nNumber of elements: {TP + TF + FF + FP}\nCorrect predictions: {TP + TF}\nIncorrect prediction: {FF + FP}\nSuccess rate: {(TP + TF)/(TP + TF + FF + FP)}')
        svm.print_time_tracking_data()
        # print('-- Prediction times: [Includes latency for internal clock]')
        #pretty_table(plaintext_data_time, ['Value index', 'Time [MS]'])
import os

if __name__ == '__main__':
    PyfhelKNN.encrypted_test()