from datetime import time
import timeit as t
import numpy as np
import csv
from tabulate import tabulate

class Timer:
    TIMEFORMAT_SEC = 1
    TIMEFORMAT_MS = 1000
    TIMEFORMAT_NS = TIMEFORMAT_MS * 1000

    def __init__(self):
        self.begin = 0
        self.end = 0
    
    def start(self):
        self.begin = 0
        self.end = 0
        self.begin = t.default_timer()

    def finish(self):
        self.end = t.default_timer()

    def get_time_in(self, time_format):
        if not (time_format == self.TIMEFORMAT_MS or self.TIMEFORMAT_NS or self.TIMEFORMAT_SEC):
            pass
        result = (self.end - self.begin) * time_format 
        self.begin = 0
        self.end = 0

        return result

def pretty_table(data_array: list, headers: list):
    print(tabulate(data_array, headers, tablefmt='orgtbl'))

def np_to_list(nparray: np.ndarray):
    return nparray.tolist()

def list_to_np(nparray: list):
    return np.asanyarray(nparray)

def minmax_normalization(data_array: list):
    min_X = min(data_array)
    max_X = max(data_array)
    frac = max_X - min_X
    normalized = [(element - min_X) / frac for element in data_array]
    return normalized

def positive_negative_noramlization(data_array: list):
    return [-1 if element <= 0 else 1 for element in data_array]

def absolute_normalization(data_array: list):
    data = [(element if element > 0 else 0) for element in data_array]
    return data 

def read_csv_to_array(path, flatten_if_possible = True):
    if not path or path == '':
        raise ValueError

    csv_file = open(path)
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = []
    
    for row in csv_reader:
        items = []
        for column in row:
            items.append(float(column))
        if len(items) == 1 and flatten_if_possible:
            data.append(items[0])
        else:
            data.append(items)

    return data

def execute_noramlization(data_array: list, normalization_type: str):
    if normalization_type == 'minmax':
        if isinstance(data_array[0], list):
            resulting_array = []
            for element in data_array:
                resulting_array.append(minmax_normalization(element))
            return resulting_array
        else:
            return minmax_normalization(data_array)

    elif normalization_type == 'positivenegative':
        if isinstance(data_array[0], list):
            resulting_array = []
            for element in data_array:
                resulting_array.append(positive_negative_noramlization(element))
            return resulting_array
        else:
            return positive_negative_noramlization(data_array)
        
    elif normalization_type == 'absolute':
        if isinstance(data_array[0], list):
            resulting_array = []
            for element in data_array:
                resulting_array.append(absolute_normalization(element))
            return resulting_array
        else:
            return absolute_normalization(data_array)
    elif normalization_type == 'none':
        return data_array

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()