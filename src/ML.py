from datetime import datetime
from utility import *
from tabulate import tabulate

class ML:
    def __init__(self, name: str) -> None:
        self.time_tracking = dict()
        self.time_tracking_names = []
        self.algorithm_name = name
        self.KEYGEN = 'keyGeneration'
        self.DATAENCX = 'inputDataEncryption'
        self.DATAENCY = 'checkDataEncryption'
        self.WENC = 'weightsEncryption'
        self.BENC = 'biasEncryption'
        self.PLAINFIT = 'plaintextFit'
        self.PLAINPREDICT = 'plaintextPredict'
        self.ENCFIT = 'encryptedFit'
        self.ENCPREDICT = 'encryptedPredict'
        self.DECDATA = 'decryptData'

        self.time_tracking_names.append(self.KEYGEN)
        self.time_tracking_names.append(self.DATAENCX)
        self.time_tracking_names.append(self.DATAENCY)
        self.time_tracking_names.append(self.WENC)
        self.time_tracking_names.append(self.BENC)
        self.time_tracking_names.append(self.PLAINFIT)
        self.time_tracking_names.append(self.PLAINPREDICT)
        self.time_tracking_names.append(self.ENCFIT)
        self.time_tracking_names.append(self.ENCPREDICT)
        self.time_tracking_names.append(self.DECDATA)

        for element in self.time_tracking_names:
            self.time_tracking[element] = -1

        self.general_timer = Timer()
        self.nested_timer = Timer()

        pass

    def encrypt_data(self, X: list) -> list:
        pass
    
    def plaintext_fit(self) -> None:
        pass

    def encrypted_fit(self) -> None:
        pass

    def plaintext_predict(self, X) -> float:
        pass

    def encrypted_predict(self, X) -> object:
        pass

    def decrypt(self, X) -> object:
        pass

    def initialize(self, X, y, cryptographic_params: dict = None, data_normalization: dict = None) -> None:
        pass
        
    def load_data(train_input_file: str, train_check_file: str, test_input_file: str, test_check_file: str):
        
        train_input_data = read_csv_to_array(train_input_file)
        train_check_data_file = read_csv_to_array(train_check_file)

        test_input_data = read_csv_to_array(test_input_file)
        test_train_check_data_file = read_csv_to_array(test_check_file)

        return train_input_data, train_check_data_file, test_input_data, test_train_check_data_file

    def get_time_tracking_data(self):
        return self.time_tracking
    
    def print_time_tracking_data(self):
        now = datetime.now()
        print('---- Time tracking ----')
        print(f'---- {self.algorithm_name} ----')
        print(f'---- {now.strftime("%d/%m/%Y %H:%M:%S")} ----')
        data = []
        for element in self.time_tracking_names:
            data.append([element, self.time_tracking[element]])
        print(tabulate(data, headers=['Name', 'Time'], tablefmt='orgtbl'))
        print('---------------')
        