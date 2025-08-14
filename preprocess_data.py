from functions import prepare_dataset, DATA_PATH
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

dirs = os.listdir(DATA_PATH)

x_trains_list = []
y_trains_list = []
x_tests_list = []
y_tests_list = []
times_tests_list = []

EXPECTED_X_DIMS = 3
EXPECTED_Y_DIMS = 2 

for i in range(4500):
    x_train, y_train, x_test, y_test, time_test, _ = prepare_dataset(dirs[i], 7, 2)

    
    if (x_train is not None and y_train is not None and x_test is not None and y_test is not None and
            x_train.ndim == EXPECTED_X_DIMS and x_test.ndim == EXPECTED_X_DIMS and
            y_train.ndim == EXPECTED_Y_DIMS and y_test.ndim == EXPECTED_Y_DIMS and
            x_train.shape[0] > 0 and x_test.shape[0] > 0):
        
        x_trains_list.append(x_train)
        y_trains_list.append(y_train)
        x_tests_list.append(x_test)
        y_tests_list.append(y_test)
        times_tests_list.append(time_test.iloc[7+1:])
    else:
        print(f"Skipping data at index {i} (file: {dirs[i]}) due to invalid shape or empty data.")


x_trains = np.concatenate(x_trains_list, axis=0)
y_trains = np.concatenate(y_trains_list, axis=0)
x_tests = np.concatenate(x_tests_list, axis=0)
y_tests = np.concatenate(y_tests_list, axis=0)
times_tests = np.concatenate(times_tests_list, axis=0)


np.savez_compressed(
    'stock_data.npz',
    x_trains=x_trains,
    y_trains=y_trains,
    x_tests=x_tests,
    y_tests=y_tests,
    times_tests=times_tests
)

print("Data successfully processed and saved to stock_data.npz")