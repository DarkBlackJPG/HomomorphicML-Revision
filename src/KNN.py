from ML import ML
import numpy as np
from utility import *
from collections import Counter

class KNN(ML):
    HAMMING = 'hamming'
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'

    def __init__(self, name: str, k: int = 2) -> None:
        super().__init__(name)
        self.k = k
        self.distance_type = 'euclidean'
    
    def hamming_distance(self, X1, X2):
	    return sum(abs(e1 - e2) for e1, e2 in zip(X1, X2)) / len(X1)

    def manhattan_distance(self, X1, X2):
        return sum(abs(e1-e2) for e1, e2 in zip(X1, X2))

    def euclidean_distance(self, X1, X2):
       return np.sqrt(sum(pow((e1 - e2),2) for e1, e2 in zip(X1, X2)))

    def encrypted_distance(self, X1, X2):
        pass

    def symmetric_encrypt(self, X) -> object:
        pass

    def symmetric_decrypt(self, X) -> object:
        pass

    def pickle_data(self, X) -> object:
        pass
    
    def change_distance_calc_type(self, distance_type):
        self.distance_type = distance_type

    def plaintext_fit(self) -> None:
        print('Implemented through initialization - no formal fitting to be done')
        pass

    def plaintext_distance(self, X1, X2):
        if self.distance_type == 'hamming':
            return self.euclidean_distance(X1, X2)
        elif self.distance_type == 'manhattan':
            return self.manhattan_distance(X1, X2)
        else:
            return self.euclidean_distance(X1, X2)

    def __predict(self, x):
        distances = [self.plaintext_distance(x, x_self) for x_self in self.X]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest = [self.y[i] for i in k_indices]
        k_nearest = [element for element in k_nearest]
        most_common = Counter(k_nearest).most_common(1)
        return most_common[0][0]

    def plaintext_predict(self, X) -> float:
        self.general_timer.start()

        if not isinstance(X[0], list):
            X = [X]
        predicted_labels = [self.__predict(x) for x in X]
        result = np.array(predicted_labels) 
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

    def plaintext_test(k = 2):
        train_input_data_file = 'data/input.csv'
        train_check_data_file = 'data/check.csv'

        test_input_data_file = 'data/input.csv' # todo
        test_check_data_file = 'data/check.csv' # todo
        
        train_input_data = read_csv_to_array(train_input_data_file)
        train_train_check_data_file = read_csv_to_array(train_check_data_file)

        test_input_data = read_csv_to_array(test_input_data_file)
        test_train_check_data_file = read_csv_to_array(test_check_data_file)

        algorithm = KNN('KNN - 2', k)
        algorithm.initialize(train_input_data, train_train_check_data_file, data_normalization= {'X': 'minmax', 'y': 'none'})
        algorithm.plaintext_fit()
        plaintext_data_time = []

        TP = 0
        TF = 0
        FP = 0
        FF = 0
        timer = Timer()
        it = 0
        for point, expected in zip(test_input_data, test_train_check_data_file):
            timer.start()
            prediction = algorithm.plaintext_predict(point)
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

        print('------' + algorithm.algorithm_name + '--------')
        print(f'TP: {TP}\nTF: {TF}\nFP: {FP}\nFF: {FF}\nNumber of elements: {TP + TF + FF + FP}\nCorrect predictions: {TP + TF}\nIncorrect prediction: {FF + FP}\nSuccess rate: {(TP + TF)/(TP + TF + FF + FP)}')
        algorithm.print_time_tracking_data()
        print('-- Prediction times: [Includes latency for internal clock]')
        pretty_table(plaintext_data_time, ['Value index', 'Time [MS]'])


if __name__ == '__main__':
    KNN.plaintext_test()