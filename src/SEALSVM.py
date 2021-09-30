from SVM import *
import numpy as np
import seal


class SEALSVM(SVM):
    def __init__(self, name: str, learning_rate, lambda_param, n_iters) -> None:
        super().__init__(name, learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
        self.cryptographic_params = None
        self.crypto_context = None
    
    def initialize(self, X, y, cryptographic_params: dict = None, data_normalization: dict = None) -> None:
        super().initialize(X, y, None, data_normalization)

        self.cryptographic_params = cryptographic_params
        if cryptographic_params is None:
            self.cryptographic_params = {}
            self.cryptographic_params['poly_modulus_degree'] = 16384
            self.cryptographic_params['coeff_modulus'] =[60, 40, 40, 60]
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
                    encoded_cell = self.ckks_encoder.encode([cell], self.scale)
                    temp_array.append(self.encryptor.encrypt(encoded_cell))
                encrypted_data.append(temp_array)
            else:
                encoded_cell = self.ckks_encoder.encode([row], self.scale)
                encrypted_data.append(self.encryptor.encrypt(encoded_cell))
        return encrypted_data
    

    def encrypted_fit(self) -> None:
        print('Homomorphic fitting not supported - Will plaintext fit then encrypt weights')
        encrypted_fit_timer = Timer()
        encrypted_fit_timer.start()

        self.plaintext_fit()

        self.nested_timer.start()
        self.encrypted_w = self.encrypt_data(np_to_list(self.w))
        self.nested_timer.finish()
        self.time_tracking[self.WENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.nested_timer.start()
        self.encrypted_b = self.encrypt_data([self.b])[0]
        self.nested_timer.finish()
        self.time_tracking[self.BENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)

        encrypted_fit_timer.finish()
        self.time_tracking[self.ENCFIT] = encrypted_fit_timer.get_time_in(Timer.TIMEFORMAT_MS)
        pass

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
        if sum_result.parms_id() != self.encrypted_b.parms_id():
            self.evaluator.mod_switch_to_inplace(sum_result, sum_result.parms_id())
            self.evaluator.mod_switch_to_inplace(self.encrypted_b, sum_result.parms_id())

        result = self.evaluator.sub(sum_result, self.encrypted_b)
        self.general_timer.finish()
        self.time_tracking[self.ENCPREDICT] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
        return result
    

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

        svm = SEALSVM('SEAL Linear SVM', learning_rate, lambda_param, n_iters)
        svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'minmax', 'y': 'none'})
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
    
if __name__ == '__main__':
    SEALSVM.encrypted_test()