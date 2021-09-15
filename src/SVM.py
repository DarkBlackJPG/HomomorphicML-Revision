from typing import Collection
from numpy.lib.function_base import select
from ML import ML
from utility import *

class SVM(ML):
    def __init__(self,name: str, learning_rate=0.001, lambda_param=0.01, n_iters=1000) -> None:
        super().__init__(name)
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def plaintext_fit(self) -> None:
        self.general_timer.start()
        
        _, n_features = self.X.shape
        y_ = np.where(self.y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(self.X):

                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

        self.general_timer.finish()
        self.time_tracking[self.PLAINFIT] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
        pass

    def plaintext_predict(self, X) -> float:
        self.general_timer.start()

        X_ = list_to_np(X)
        approx = np.dot(X_, self.w) - self.b
        result = np.sign(approx)

        self.general_timer.finish()
        self.time_tracking[self.PLAINPREDICT] = self.general_timer.get_time_in(Timer.TIMEFORMAT_MS)
        return result

    def initialize(self, X, y, cryptographic_params: dict = None, data_normalization: dict = None) -> None:
        if cryptographic_params:
            print('Crypto params are not used')

        self.X = X
        self.y = y

        if data_normalization:
            normalization_type_for_X = data_normalization['X']
            normalization_type_for_y = data_normalization['y']

            self.X = execute_noramlization(X, normalization_type_for_X)
            self.y = execute_noramlization(y, normalization_type_for_y)

        self.X = list_to_np(self.X)
        self.y = list_to_np(self.y)
        self.n_samples, self.n_features = self.X.shape
        self.nextPowerOfTwo = next_power_of_2(self.n_features + 1)

    def plaintext_test(learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data('data/input.csv', 'data/check.csv', 'data/input.csv', 'data/check.csv')

        svm = SVM('Linear SVM', learning_rate, lambda_param, n_iters)
        svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'minmax', 'y': 'none'})
        svm.plaintext_fit()
        plaintext_data_time = []

        TP = 0
        TF = 0
        FP = 0
        FF = 0
        timer = Timer()
        it = 0
        for point, expected in zip(test_input_data, test_train_check_data_file):
            timer.start()
            prediction = svm.plaintext_predict(point)
            timer.finish()
            plaintext_data_time.append([it, timer.get_time_in(Timer.TIMEFORMAT_MS)])
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

            it += 1

        print('------' + svm.algorithm_name + '--------')
        print(f'TP: {TP}\nTF: {TF}\nFP: {FP}\nFF: {FF}\nNumber of elements: {TP + TF + FF + FP}\nCorrect predictions: {TP + TF}\nIncorrect prediction: {FF + FP}\nSuccess rate: {(TP + TF)/(TP + TF + FF + FP)}')
        svm.print_time_tracking_data()
        print('-- Prediction times: [Includes latency for internal clock]')
        pretty_table(plaintext_data_time, ['Value index', 'Time [MS]'])


if __name__ == '__main__':
    SVM.plaintext_test()
        
