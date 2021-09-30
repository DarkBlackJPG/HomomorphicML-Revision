from SVM import *
from Pyfhel import *

class PyfhelSVM(SVM):
    def __init__(self, name: str, learning_rate, lambda_param, n_iters) -> None:
        super().__init__(name, learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
        self.cryptographic_params = None
        self.crypto_context = None
    
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

        
        self.encrypted_b = self.encrypt_data(self.b)

        self.nested_timer.start()
        self.encrypted_w = self.encrypt_data(np_to_list(self.w))
        self.nested_timer.finish()
        self.time_tracking[self.WENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.nested_timer.start()
        self.encrypted_b = self.encrypt_data(self.b)
        self.nested_timer.finish()
        self.time_tracking[self.BENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)

        encrypted_fit_timer.finish()
        self.time_tracking[self.ENCFIT] = encrypted_fit_timer.get_time_in(Timer.TIMEFORMAT_MS)
        pass

    def encrypted_predict(self, X) -> object:
        self.general_timer.start()

        is_encrypted = isinstance(X[0], PyCtxt)
        X_ = X
        if not is_encrypted:
            X_ = self.encrypt_data(X)
        
        encrypted_mult = [x_ * w_ for x_, w_ in zip(X_, self.encrypted_w)]
        encrypted_sum = self.crypto_context.encryptFrac(0)
        for element in encrypted_mult:
            encrypted_sum = self.crypto_context.add(element, encrypted_sum, True)
        
        result = self.crypto_context.sub(encrypted_sum, self.encrypted_b, True)

        self.general_timer.finish()
        self.time_tracking[self.ENCPREDICT] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
        return result
    

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

    def encrypted_test(learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data('data/input.csv', 'data/check.csv', 'data/input.csv', 'data/check.csv')

        svm = PyfhelSVM('Linear SVM', learning_rate, lambda_param, n_iters)
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
    PyfhelSVM.encrypted_test()