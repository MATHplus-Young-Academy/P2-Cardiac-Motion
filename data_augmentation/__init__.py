'''
First Version: 17.03.2022
Author Darian Viezzer, David
Copyright 2022 Charité Universitätsmedizin Berlin
Copyright 2022
'''

import pydicom
import numpy as np
import copy

def augmentation(image, probability_limit=25, show_array=False):
    array = copy.copy(image)
    #array = copy.deepcopy(image).astype("float32")
    min_val = np.min(array)
    max_val = np.max(array)
    array = array / max_val
    
    if show_array:
        print("INPUT")
        print(array)
    
    # brightness
    prob=np.random.randint(0, 100)
    if prob<probability_limit:
        excitation = 1 + (np.clip(np.random.normal(0.0, 0.388), -1, 1) * 0.5)
        array = excitation * array

    if show_array:
        print("BRIGTHNESS")
        print(array)

    # contrast
    prob=np.random.randint(0, 100)
    if prob<probability_limit:
        change_val = (prob / 1000)
        prob=np.random.randint(0, 2)
        if prob < 1:
            change_val = -change_val
        array = array + change_val
    
    if show_array:
        print("CONTRAST")
        print(array)

    # blurring:
    prob=np.random.randint(0, 100)
    if prob < probability_limit:
        k_power = float((abs(np.clip(np.random.normal(0.0, 0.388), -1, 1)) + 1) / 2)
        k_con = k_power * np.array([0.5, 1, 0.5])
        array = np.apply_along_axis(lambda x: np.convolve(x, k_con, mode='same'), 0, array)
        array = np.apply_along_axis(lambda x: np.convolve(x, k_con, mode='same'), 1, array)
    
    if show_array:
        print("BLURRING")
        print(array)

    # add noise
    prob=np.random.randint(0, 100)
    if prob<probability_limit: #gaussian random noise
        std = abs(np.clip(np.random.normal(0.0, 0.388), -1, 1))
        if std != 0:
            noise = np.clip(np.random.normal(0, std, np.shape(array)), -2, 2) * (1 / 100)
            array = array + noise
    if show_array:
        print("NOISE - GAUSSIAN")
        print(array)

    prob=np.random.randint(0, 100)
    if prob<probability_limit: # uniform noise
        bins = int((abs(np.clip(np.random.normal(0.0, 0.388), -1, 1)) + 1) * 500)
        if bins > 100:
            array = np.digitize(array, np.linspace(np.min(array), np.max(array), bins)) / bins
    
    if show_array:
        print("NOISE - UNIFORM")
        print(array)

    
    prob=np.random.randint(0, 100)
    if prob<probability_limit: # salt and pepper
        row,col = np.shape(array)
        sp = 0.5
        amount = 0.0005 * (abs(np.clip(np.random.normal(0.0, 0.388), -1, 1)) + 1) * 50
        # Salt mode
        num_salt = np.ceil(amount * row * col * sp)
        coords_x = np.random.randint(0, row, int(num_salt))
        coords_y = np.random.randint(0, col, int(num_salt))
        array[coords_x, coords_y] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * row * col * (1 - sp))
        coords_x = np.random.randint(0, row, int(num_pepper))
        coords_y = np.random.randint(0, col, int(num_pepper))
        array[coords_x, coords_y] = 0

    if show_array:
        print("NOISE - SALT AND PEPPER")
        print(array)

    array[array>1] = 1
    array[array<0] = 0
    
    if show_array:
        print("OUTPUT")
        print(array)

    return array

if __name__ == "__main__":
    image = 10 * np.random.rand(10,10)
    test = augmentation(image, 100, True)