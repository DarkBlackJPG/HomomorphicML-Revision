import pickle
from KNN import KNN
import seal
from KNN import *
from AES import *
import random
from DataSender import DataSender
from DataReceiver import DataReceiver
import time

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
                    encoded_cell = self.ckks_encoder.encode(cell, self.scale)
                    temp_array.append(self.encryptor.encrypt(encoded_cell))
                encrypted_data.append(temp_array)
            else:
                encoded_cell = self.ckks_encoder.encode(row, self.scale)
                encrypted_data.append(self.encryptor.encrypt(encoded_cell))
        return encrypted_data
    

    def encrypted_fit(self) -> None:
        print('Homomorphic fitting not supported - Will plaintext fit then encrypt weights')
        encrypted_fit_timer = Timer()
        encrypted_fit_timer.start()

        self.plaintext_fit()

        self.encrypted_w = self.encrypt_data(np_to_list(self.w))
        self.encrypted_b = self.encrypt_data(self.b)

        encrypted_fit_timer.finish()
        self.time_tracking[self.ENCFIT] = encrypted_fit_timer.get_time_in(Timer.TIMEFORMAT_MS)
        pass

    
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

    

    def encrypted_test(learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data('data/input.csv', 'data/check.csv', 'data/input.csv', 'data/check.csv')

        svm = SEALKNN('SEAL KNN', 2)
        svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'minmax', 'y': 'none'})
        svm.confuse_data([1,2,3], '123')
        svm.encrypted_fit()
        print(np.sign(svm.decrypt(svm.encrypted_predict(np_to_list(svm.X[23])))), svm.y[23])
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

        is_encrypted = isinstance(X[0], seal.Ciphertext)
        X_ = X
        if not is_encrypted:
            X_ = self.encrypt_data([X])[0]

        temp_result_array = []
        for x, y in zip(X_, self.encrypted_w):
            result = self.evaluator.multiply(x, y)
            self.evaluator.relinearize_inplace(result, self.relin_key)
            self.evaluator.rescale_to_next_inplace(result)
            result.scale(self.scale)
            temp_result_array.append(result)
        
        sum_result = self.evaluator.add_many(temp_result_array)
        # ToDo Sta ovo radi sunce ti jebem
        if sum_result.parms_id() != self.encrypted_b.parms_id():
                self.evaluator.mod_switch_to_inplace(sum_result, sum_result.parms_id())
                self.evaluator.mod_switch_to_inplace(self.encrypted_b, sum_result.parms_id())
        result = self.evaluator.add(sum_result, self.encrypted_b)
        
        self.general_timer.finish()
        self.time_tracking[self.ENCPREDICT] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
        return result
    
    
    def seal_single_encrypt(self, data):
        encoded_data = self.ckks_encoder.encode(data, self.scale)
        return self.encryptor.encrypt(encoded_data)

    def seal_add(self, first, second):
        if first.parms_id() != second.parms_id():
                self.evaluator.mod_switch_to_inplace(first, first.parms_id())
                self.evaluator.mod_switch_to_inplace(second, first.parms_id())
        result = self.evaluator.add(first, second)

        return result
    
    def confuse_data(self, X, password: str):
        random.seed(int(time.time()))
        self.confusion_random_value = random.random()
        self.permutation = [i for i in range(0, len(X))]
        random.shuffle(self.permutation)
        aes = AES()

        return_array = []
        for i in range(0, len(X)):
            temporary_confused_data = self.seal_add(seal.Ciphertext(X[i]), self.seal_single_encrypt(self.confusion_random_value))
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


    
if __name__ == '__main__':
    SEALKNN.encrypted_test()