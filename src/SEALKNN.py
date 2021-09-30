import pickle

from numpy import bytes0
from KNN import KNN
import seal
from KNN import *
from AES import *
import random
from DataSender import DataSender
from DataReceiver import DataReceiver
import time
import os
import math

class SEALKNN(KNN):
    def __init__(self, name: str, k: int = 2) -> None:
        super().__init__(name, k)
        self.cryptographic_params = None
        self.crypto_context = None
        self.password = 'XDK123XDK'
        self.disclosed_iv = b'\xc8\x8b\xf7\rn\xaf.\xd67\xb0\xd8\xd7\xd8\x17\x1cf'
    
    def initialize(self, X, y, cryptographic_params: dict = None, data_normalization: dict = None) -> None:
        super().initialize(X, y, None, data_normalization)

        self.cryptographic_params = cryptographic_params
        if cryptographic_params is None:
            self.cryptographic_params = {}
            self.cryptographic_params['poly_modulus_degree'] = 16384
            self.cryptographic_params['coeff_modulus'] = [60, 40, 40, 40, 40, 60]
            self.cryptographic_params['scale'] = 2.0 ** 40
        
        self.general_timer.start()
        

        self.scale = self.cryptographic_params['scale']
        self.params = seal.EncryptionParameters(seal.scheme_type.ckks)
        self.params.set_poly_modulus_degree(self.cryptographic_params['poly_modulus_degree'])
        self.params.set_coeff_modulus(seal.CoeffModulus.Create(self.cryptographic_params['poly_modulus_degree'], self.cryptographic_params['coeff_modulus']))
        self.context = seal.SEALContext(self.params)
        self.ckks_encoder = seal.CKKSEncoder(self.context)
        self.slot_count = self.ckks_encoder.slot_count()
        self.keygen = seal.KeyGenerator(self.context)
        self.public_key = self.keygen.create_public_key()
        self.secret_key = self.keygen.secret_key()
        self.relin_key = self.keygen.create_relin_keys()

        self.encryptor = seal.Encryptor(self.context, self.public_key)
        self.evaluator = seal.Evaluator(self.context)
        self.decryptor = seal.Decryptor(self.context, self.secret_key)
        self.general_timer.finish()
        self.time_tracking[self.KEYGEN] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)

        temp_data = {}
        temp_data['scale'] = self.scale
        temp_data['params'] = self.params
        temp_data['context'] = self.context
        temp_data['ckks_encoder'] = self.ckks_encoder
        temp_data['slot_count'] = self.slot_count
        temp_data['keygen'] = self.keygen
        temp_data['encryptor'] = self.encryptor
        temp_data['evaluator'] = self.evaluator
        temp_data['decryptor'] = self.decryptor

        self.general_timer.start()
        self.encrypted_X = self.encrypt_data(np_to_list(self.X))
        self.general_timer.finish()
        self.time_tracking[self.DATAENCX] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.general_timer.start()
        self.encrypted_y = self.encrypt_data(np_to_list(self.y))
        self.general_timer.finish()
        self.time_tracking[self.DATAENCY] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)

    def encrypt_data(self, X) -> object:
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            result = self.ckks_encoder.encode(X, self.scale)
            return self.encryptor.encrypt(result)

        encrypted_data = []
        for row in X:
            if isinstance(row, list) and len(row) > 1: 
                temp_array = []
                for cell in row:
                    temp_array.append(self.seal_single_encrypt(cell))
                encrypted_data.append(temp_array)
            else:
                encrypted_data.append(self.seal_single_encrypt(row))
        return encrypted_data
    

    def encrypted_fit(self) -> None:
        print('Homomorphic fitting not supported - Will plaintext fit then encrypt weights')
        encrypted_fit_timer = Timer()
        encrypted_fit_timer.start()

        self.plaintext_fit()

        encrypted_fit_timer.finish()
        self.time_tracking[self.ENCFIT] = encrypted_fit_timer.get_time_in(Timer.TIMEFORMAT_MS)

    
    def decrypt(self, X) -> object:
        self.general_timer.start()
        result = None
        if isinstance(X, list):
            print('Expected ciphertext object, got list')
            raise ValueError
        else:
            result = self.decryptor.decrypt(X)
            result = self.ckks_encoder.decode(result)
        
        self.general_timer.finish()
        self.time_tracking[self.DECDATA] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS) 
        return result

    

    def encrypted_test(k = 2):
        train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data('data/simple_input_test.csv', 'data/simple_check_test.csv', 'data/simple_input_test.csv', 'data/simple_check_test.csv')

        svm = SEALKNN('SEAL KNN', k)
        svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'minmax', 'y': 'none'})
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



    def encrypted_predict(self, X) -> object:
        self.general_timer.start()

        if len(X) > 1:
            print('Expected only one point. No batching! shape(1, <number_features>)')
            pass
        
        encrypted_X = X[0]

        is_encrypted = isinstance(encrypted_X[0], seal.Ciphertext)
        if not is_encrypted:
            encrypted_X = self.encrypt_data(encrypted_X)
        

        if len(encrypted_X) != self.n_features:
            print(f'Incorrect size of data. Expected {self.n_features} features')
            raise ValueError

        # Calculate distance
        encrypted_distance = []
        for my_data in self.encrypted_X:
            encrypted_distance.append(self.encrypted_distance(encrypted_X, my_data))

        confused_data = self.confuse_data(encrypted_distance, self.password)

        self.general_timer.finish()
        self.time_tracking[self.ENCPREDICT] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
        return confused_data
    
    
    def seal_single_encrypt(self, data):
        encoded_data = self.ckks_encoder.encode(data, self.scale)
        return self.encryptor.encrypt(encoded_data)
    
    def rescale(self, first, second):
        first = self.evaluator.rescale_to(first, first.parms_id())
        second = self.evaluator.rescale_to(second, first.parms_id())

        return first, second


    def decrypt_decode_(self, data):
        decrypted = self.decryptor.decrypt(data)
        decoded = self.ckks_encoder.decode(decrypted)
        return decoded
    
    def confuse_data(self, X, password: str):
        return_array = []
        for i in range(0, len(X)):
            return_array.append((X[i], self.encrypted_y[i]))
    
        
        return return_array

    def read_bytes_from_file(self, filename: str, remove = False):
        file = open(filename, "rb")
        bytes = file.read()
        file.close()
        if remove:
            os.remove(filename)
        return bytes

    def get_HE_context_bytes(self):
        context_dict = dict()
        
        self.public_key.save('public_key')
        self.params.save('params')
        self.secret_key.save('secret_key')
        self.relin_key.save('relin_keys')



        context_dict['scale'] = self.scale 
        context_dict['slot_count'] = self.slot_count
        context_dict['public_key'] = self.read_bytes_from_file('public_key', True)
        context_dict['params'] = self.read_bytes_from_file('params', True)
        context_dict['secret_key'] = self.read_bytes_from_file('secret_key', True)
        context_dict['relin_keys'] = self.read_bytes_from_file('relin_keys', True)

        context_bytes = pickle.dumps(context_dict)
        
        return context_bytes

    def send_data_and_return_sorted(self, data, additional_data: dict = None):
      pass
    
    def encrypted_distance(self, X1, X2):
        X_1 = [seal.Ciphertext(x) for x in X1] # Copy
        X_2 = [seal.Ciphertext(x) for x in X2] # Copy
        encrypted_sum = None
        for x1_element, x2_element in zip(X_1, X_2):
             
            if x1_element.parms_id() != x2_element.parms_id():
                self.evaluator.mod_switch_to_inplace(x1_element, x1_element.parms_id())
                self.evaluator.mod_switch_to_inplace(x2_element, x1_element.parms_id())

            if not encrypted_sum:
                encrypted_sum = self.evaluator.sub(x1_element, x2_element)
                encrypted_sum = self.evaluator.square(encrypted_sum)
                encrypted_sum = self.evaluator.relinearize(encrypted_sum, self.relin_key)
            else:
                temp_subtraction = self.evaluator.sub(x1_element, x2_element)
                square = self.evaluator.square(temp_subtraction)
                square = self.evaluator.relinearize(square, self.relin_key)

                if square.parms_id() != encrypted_sum.parms_id():
                    self.evaluator.mod_switch_to_inplace(square, square.parms_id())
                    self.evaluator.mod_switch_to_inplace(encrypted_sum, square.parms_id())

                encrypted_sum = self.evaluator.add(square, encrypted_sum)
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


    
if __name__ == '__main__':
    SEALKNN.encrypted_test()