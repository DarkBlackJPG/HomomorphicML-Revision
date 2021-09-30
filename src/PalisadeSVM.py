from numpy.lib.arraysetops import isin
from sklearn import utils
from SVM import *
import pycrypto
from utility import execute_noramlization
import pandas as pd

class PalisadeSVM(SVM):
    def __init__(self, name: str, learning_rate, lambda_param, n_iters) -> None:
        super().__init__(name, learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
        self.cryptographic_params = None
        self.crypto_context = None
    
    def encrypt_data(self, X) -> object:
        if isinstance(X, np.ndarray):
            raise ValueError

        if not isinstance(X, list):
            return self.crypto_context.encryptFrac(X)

        encrypted_data = []
        for row in X:
            if isinstance(row, list): # and len(row) > 1: 
                encrypted_data.append(self.crypto_context.Encrypt(row))
            else:
                print('Invalid data for encryption, must be list')
                raise ValueError
        return encrypted_data
    

    def encrypted_fit(self) -> None:
        print('Homomorphic fitting not supported - Will plaintext fit then encrypt weights')
        encrypted_fit_timer = Timer()
        encrypted_fit_timer.start()

        self.plaintext_fit()

        self.nested_timer.start()
        self.encrypted_w = self.encrypt_data([np_to_list(self.w)])[0]
        self.nested_timer.finish()
        self.time_tracking[self.WENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.nested_timer.start()
        self.encrypted_b = self.encrypt_data([[self.b] * self.n_features])[0] # Must be size of data
        self.nested_timer.finish()
        self.time_tracking[self.BENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)

        encrypted_fit_timer.finish()
        self.time_tracking[self.ENCFIT] = encrypted_fit_timer.get_time_in(Timer.TIMEFORMAT_MS)
        pass

    def encrypted_predict(self, X) -> object:
        self.general_timer.start()

        is_encrypted = isinstance(X[0], pycrypto.Ciphertext)
        X_ = X
        if not is_encrypted:
            X_ = self.encrypt_data([X])[0]

        # Todo
        result = self.crypto_context.EvalMultAndRelinearize(X_, self.encrypted_w)
        result = self.encrypt_data([self.decrypt(result)])[0]
        result = self.crypto_context.EvalSum(result, self.nextPowerOfTwo)
        result = self.crypto_context.EvalSub(result, self.encrypted_b)

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
            result = self.crypto_context.Decrypt(X)
        
        self.general_timer.finish()
        self.time_tracking[self.DECDATA] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS) 
        return result

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

    def encrypted_test(learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        dataset = 'simple'
        train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data(
            'data/'+dataset+'_input_train.csv', 'data/'+dataset+'_check_train.csv',
            'data/'+dataset+'_input_test.csv', 'data/'+dataset+'_check_test.csv'
            )

        svm = PalisadeSVM('Palisade Linear SVM', learning_rate, lambda_param, n_iters)
        svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'minmax', 'y': 'none'})
        svm.encrypted_fit()
        plaintext_data_time = []

        TP = 0
        TF = 0
        FP = 0
        FF = 0

        TPE = 0
        TFE = 0
        FPE = 0
        FFE = 0
        timer = Timer()
        it = 0
        encrypted_prediction = []
        expectedd = []
        plaintext_prediction = []
        
        for point, expected in zip(test_input_data, test_train_check_data_file):
            expectedd.append(expectedd)
            point = execute_noramlization(point, 'minmax')
            prediction = svm.plaintext_predict(point)
            plaintext_prediction.append(prediction)
            encrypted_prediction = np.sign(svm.decrypt(svm.encrypted_predict(point)))[0]
            plaintext_prediction.append(encrypted_prediction)

            if prediction == expected:
                if prediction > 0:
                    TP += 1
                else:
                    TF += 1
            else: 
                if prediction > 0:
                    FP += 1
                else:
                    FF += 1

            if encrypted_prediction == expected:
                if encrypted_prediction > 0:
                    TPE += 1
                else:
                    TFE += 1
            else: 
                if encrypted_prediction > 0:
                    FPE += 1
                else:
                    FFE += 1

            it += 1

        print('------' + svm.algorithm_name + '--------')
        print(f'TP: {TP}\nTF: {TF}\nFP: {FP}\nFF: {FF}\nNumber of elements: {TP + TF + FF + FP}\nCorrect predictions: {TP + TF}\nIncorrect prediction: {FF + FP}\nSuccess rate: {(TP + TF)/(TP + TF + FF + FP)}')
        print('---- Encrypted statistic ---- ')
        print(f'TP: {TPE}\nTF: {TFE}\nFP: {FPE}\nFF: {FFE}\nNumber of elements: {TPE + TFE + FFE + FPE}\nCorrect predictions: {TPE + TFE}\nIncorrect prediction: {FFE + FPE}\nSuccess rate: {(TPE + TFE)/(TPE + TFE + FFE + FPE)}')
        svm.print_time_tracking_data()
        print('-- Prediction times: [Includes latency for internal clock]')

    
if __name__ == '__main__':
    PalisadeSVM.encrypted_test()