from scipy.sparse import data
from PalisadeSVM import *
from PyfhelKNN import *
from SEALKNN import SEALKNN
from SEALSVM import *
from PalisadeKNN import PalisadeKNN
from PyfhelSVM import *
from collections import Counter
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
import pandas as pd
import threading
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sn
import gc
from utility import Timer
cpu = [0]
ram = [0]
time_passed = [1] # ms
lck = threading.Lock()

def utility():
    global cpu
    global ram
    global time_passed
    while True:
        time.sleep(0.25)
        lck.acquire()
        cpu[0] += psutil.cpu_percent()
        ram[0] += psutil.virtual_memory().percent
        time_passed[0] += 1
        lck.release()
    
datasets = ['simple', 'ion', 'credit']
learning_rate = 0.001
lambda_param = 0.01
n_iters = 1000

x = threading.Thread(target=utility, args=())
x.start()

doKNNCredit = False
shouldSkipKNNCredit = False
timer_ = Timer()


if not doKNNCredit:

    for dataset in datasets:
        timer_.start()
        algorithm_name = 'Palisade_Linear_SVM'

        lck.acquire()
        cpu = [0]
        ram = [0]
        time_passed = [0.0000000000000001] # ms
        lck.release()


        print (f'------ DATASET: {dataset} ------')
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

        for point, expected in zip(test_input_data, test_train_check_data_file):
    
            point = execute_noramlization(point, 'minmax')

            prediction = svm.plaintext_predict(point)
            encrypted_prediction = np.sign(svm.decrypt(svm.encrypted_predict(point)))[0]

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
        timer_.finish()
        print(algorithm_name+'_'+dataset +':  '+ str(timer_.get_time_in(timer_.TIMEFORMAT_SEC)))

        lck.acquire()
        cpu_usage = cpu[0]
        ram_usage = ram[0]
        time_passed_in_Ms = time_passed[0]
        lck.release()


        print(f'Avg CPU Usage: {cpu_usage/time_passed_in_Ms} ')
        print(f'Avg RAM Usage: {ram_usage/time_passed_in_Ms} ')

        plain = [[TP, TF], [FP, FF]]
        enc = [[TPE, TFE], [FPE, FFE]]
        plain_dataframe = pd.DataFrame(plain, range(2), range(2))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(plain_dataframe, annot=True, annot_kws={"size": 16}) # font size
        plt.title(algorithm_name + "_" + dataset)
        plt.savefig('results/'+algorithm_name+'_'+dataset+'.jpg')
        plt.clf()
        del svm
        gc.collect()

    for dataset in datasets:
        timer_.start()
        algorithm_name = 'SEAL_Linear_SVM'

        lck.acquire()
        cpu = [0]
        ram = [0]
        time_passed = [0.0000000000000001] # ms
        lck.release()


        print (f'------ DATASET: {dataset} ------')
        train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data(
            'data/'+dataset+'_input_train.csv', 'data/'+dataset+'_check_train.csv',
            'data/'+dataset+'_input_test.csv', 'data/'+dataset+'_check_test.csv'
            )

        svm = SEALSVM('SEAL Linear SVM', learning_rate, lambda_param, n_iters)
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

        for point, expected in zip(test_input_data, test_train_check_data_file):
            
            point = execute_noramlization(point, 'minmax')
            prediction = svm.plaintext_predict(point)
            decrypted = svm.decrypt(svm.encrypted_predict(point))
            encrypted_prediction = np.sign(decrypted)[0]

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
                #print(decrypted, expected)
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
        timer_.finish()
        print(algorithm_name+'_'+dataset + str(timer_.get_time_in(timer_.TIMEFORMAT_MS)))
        lck.acquire()
        cpu_usage = cpu[0]
        ram_usage = ram[0]
        time_passed_in_Ms = time_passed[0]
        lck.release()


        print(f'Avg CPU Usage: {cpu_usage/time_passed_in_Ms} ')
        print(f'Avg RAM Usage: {ram_usage/time_passed_in_Ms} ')

        plain = [[TP, TF], [FP, FF]]
        enc = [[TPE, TFE], [FPE, FFE]]
        plain_dataframe = pd.DataFrame(plain, range(2), range(2))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(plain_dataframe, annot=True, annot_kws={"size": 16}) # font size
        plt.title(algorithm_name + "_" + dataset)
        plt.savefig('results/'+algorithm_name+'_'+dataset+'.jpg')
        plt.clf()
        del svm
        gc.collect()

    for dataset in datasets:
        timer_.start()
        algorithm_name = 'Pyfhel_Linear_SVM'

        lck.acquire()
        cpu = [0]
        ram = [0]
        time_passed = [0.0000000000000001] # ms
        lck.release()
        print (f'------ DATASET: {dataset} ------')
        train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data(
            'data/'+dataset+'_input_train.csv', 'data/'+dataset+'_check_train.csv',
            'data/'+dataset+'_input_test.csv', 'data/'+dataset+'_check_test.csv'
            )

        svm = PyfhelSVM('Pyfhel Linear SVM', learning_rate, lambda_param, n_iters)
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

        for point, expected in zip(test_input_data, test_train_check_data_file):
            
            point = execute_noramlization(point, 'minmax')
            prediction = svm.plaintext_predict(point)
            
            encrypted_prediction = np.sign(svm.decrypt(svm.encrypted_predict(point)))

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
        timer_.finish()
        print(algorithm_name+'_'+dataset + str(timer_.get_time_in(timer_.TIMEFORMAT_MS)))
        lck.acquire()
        cpu_usage = cpu[0]
        ram_usage = ram[0]
        time_passed_in_Ms = time_passed[0]
        lck.release()


        print(f'Avg CPU Usage: {cpu_usage/time_passed_in_Ms} ')
        print(f'Avg RAM Usage: {ram_usage/time_passed_in_Ms} ')

        plain = [[TP, TF], [FP, FF]]
        enc = [[TPE, TFE], [FPE, FFE]]
        plain_dataframe = pd.DataFrame(plain, range(2), range(2))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(plain_dataframe, annot=True, annot_kws={"size": 16}) # font size
        plt.title(algorithm_name + "_" + dataset)
        plt.savefig('results/'+algorithm_name+'_'+dataset+'.jpg')
        plt.clf()
        del svm
        gc.collect()


for dataset in datasets:

    if shouldSkipKNNCredit and dataset == 'credit':
        continue

    timer_.start()
    algorithm_name = 'Palisade KNN'

    lck.acquire()
    cpu = [0]
    ram = [0]
    time_passed = [0.0000000000000001] # ms
    lck.release()


    print (f'------ DATASET: {dataset} ------')
    train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data(
        'data/'+dataset+'_input_train.csv', 'data/'+dataset+'_check_train.csv',
        'data/'+dataset+'_input_test.csv', 'data/'+dataset+'_check_test.csv'
        )
    k = 2
    svm = PalisadeKNN('Palisade KNN', k)
    svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'none', 'y': 'none'})
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

    for point, expected in zip(test_input_data, test_train_check_data_file):
        
        #point = execute_noramlization(point, 'minmax')
        prediction = svm.plaintext_predict(point)
        prediction =  Counter(prediction).most_common(1)
        prediction = prediction[0]
        encrypted_prediction = svm.encrypted_predict([point])
        decrypted_array = []
        for element in encrypted_prediction:
            decrypted_array.append((svm.decrypt(element[0])[0], svm.decrypt(element[1])[0] ))
        sorted_data = sorted(decrypted_array, key=lambda tup: tup[0])[0:k]
        sorted_data = [element[1] for element in sorted_data]
        encrypted_prediction = Counter(sorted_data).most_common(1)[0]
        
        encrypted_prediction = np.sign(encrypted_prediction[0])

        if prediction[0] == expected:
            if prediction[0] > 0:
                TP += 1
            else:
                TF += 1
        else: 
            if prediction[0] > 0:
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
    timer_.finish()
    print(algorithm_name+'_'+dataset + str(timer_.get_time_in(timer_.TIMEFORMAT_MS)))
    lck.acquire()
    cpu_usage = cpu[0]
    ram_usage = ram[0]
    time_passed_in_Ms = time_passed[0]
    lck.release()


    print(f'Avg CPU Usage: {cpu_usage/time_passed_in_Ms} ')
    print(f'Avg RAM Usage: {ram_usage/time_passed_in_Ms} ')

    plain = [[TP, TF], [FP, FF]]
    enc = [[TPE, TFE], [FPE, FFE]]
    plain_dataframe = pd.DataFrame(plain, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(plain_dataframe, annot=True, annot_kws={"size": 16}) # font size
    plt.title(algorithm_name + "_" + dataset)
    plt.savefig('results/'+algorithm_name+'_'+dataset+'.jpg')
    plt.clf()
    del svm
    gc.collect()

for dataset in datasets:
    if shouldSkipKNNCredit and dataset == 'credit':
        continue
    timer_.start()
    algorithm_name = 'SEAL_KNN'

    lck.acquire()
    cpu = [0]
    ram = [0]
    time_passed = [0.0000000000000001] # ms
    lck.release()
    print (f'------ DATASET: {dataset} ------')
    train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data(
        'data/'+dataset+'_input_train.csv', 'data/'+dataset+'_check_train.csv',
        'data/'+dataset+'_input_test.csv', 'data/'+dataset+'_check_test.csv'
        )
    k = 2
    svm = SEALKNN('SEAL KNN', k)
    svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'none', 'y': 'none'})
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

    for point, expected in zip(test_input_data, test_train_check_data_file):
        
        #point = execute_noramlization(point, 'minmax')
        prediction = svm.plaintext_predict(point)
        prediction =  Counter(prediction).most_common(1)
        prediction = prediction[0]
        encrypted_prediction = svm.encrypted_predict([point])
        decrypted_array = [(svm.decrypt(dist)[0], svm.decrypt(Klass)[0]) for dist, Klass in encrypted_prediction]
        sorted_data = sorted(decrypted_array, key=lambda tup: tup[0])[0:k]
        sorted_data = [element[1] for element in sorted_data]
        encrypted_prediction = Counter(sorted_data).most_common(1)[0]
        
        encrypted_prediction = np.sign(encrypted_prediction[0])

        if prediction[0] == expected:
            if prediction[0] > 0:
                TP += 1
            else:
                TF += 1
        else: 
            if prediction[0] > 0:
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
    timer_.finish()
    print(algorithm_name+'_'+dataset + str(timer_.get_time_in(timer_.TIMEFORMAT_MS)))
    lck.acquire()
    cpu_usage = cpu[0]
    ram_usage = ram[0]
    time_passed_in_Ms = time_passed[0]
    lck.release()


    print(f'Avg CPU Usage: {cpu_usage/time_passed_in_Ms} ')
    print(f'Avg RAM Usage: {ram_usage/time_passed_in_Ms} ')

    plain = [[TP, TF], [FP, FF]]
    enc = [[TPE, TFE], [FPE, FFE]]
    plain_dataframe = pd.DataFrame(plain, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(plain_dataframe, annot=True, annot_kws={"size": 16}) # font size
    plt.title(algorithm_name + "_" + dataset)
    plt.savefig('results/'+algorithm_name+'_'+dataset+'.jpg')
    plt.clf()
    del svm
    gc.collect()



for dataset in datasets:

    if shouldSkipKNNCredit and dataset == 'credit':
        continue
    
    timer_.start()

    algorithm_name = 'Pyfhel_KNN'

    lck.acquire()
    cpu = [0]
    ram = [0]
    time_passed = [0.0000000000000001] # ms
    lck.release()

    print (f'------ DATASET: {dataset} ------')
    train_input_data, train_check_data_file, test_input_data, test_train_check_data_file = ML.load_data(
        'data/'+dataset+'_input_train.csv', 'data/'+dataset+'_check_train.csv',
        'data/'+dataset+'_input_test.csv', 'data/'+dataset+'_check_test.csv'
        )
    k = 2
    svm = PyfhelKNN('Pyfhel KNN', k)
    svm.initialize(train_input_data, train_check_data_file, data_normalization= {'X': 'none', 'y': 'none'})
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

    for point, expected in zip(test_input_data, test_train_check_data_file):
        
        #point = execute_noramlization(point, 'minmax')
        prediction = svm.plaintext_predict(point)
        prediction =  Counter(prediction).most_common(1)
        prediction = prediction[0]
        encrypted_prediction = svm.encrypted_predict([point])
        decrypted_array = [(svm.decrypt(dist), svm.decrypt(Klass)) for dist, Klass in encrypted_prediction]
        sorted_data = sorted(decrypted_array, key=lambda tup: tup[0])[0:k]
        sorted_data = [element[1] for element in sorted_data]
        encrypted_prediction = Counter(sorted_data).most_common(1)[0]
        
        encrypted_prediction = encrypted_prediction[0]

        if prediction[0] == expected:
            if prediction[0] > 0:
                TP += 1
            else:
                TF += 1
        else: 
            if prediction[0] > 0:
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
    timer_.finish()
    print(algorithm_name+'_'+dataset + str(timer_.get_time_in(timer_.TIMEFORMAT_MS)))
    lck.acquire()
    cpu_usage = cpu[0]
    ram_usage = ram[0]
    time_passed_in_Ms = time_passed[0]
    lck.release()


    print(f'Avg CPU Usage: {cpu_usage/time_passed_in_Ms} ')
    print(f'Avg RAM Usage: {ram_usage/time_passed_in_Ms} ')

    plain = [[TP, TF], [FP, FF]]
    enc = [[TPE, TFE], [FPE, FFE]]
    plain_dataframe = pd.DataFrame(plain, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(plain_dataframe, annot=True, annot_kws={"size": 16}) # font size
    plt.title(algorithm_name + "_" + dataset)
    plt.savefig('results/'+algorithm_name+'_'+dataset+'.jpg')
    plt.clf()
    del svm
    gc.collect()