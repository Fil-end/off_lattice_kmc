import numpy as np

def to_pad_the_array(array, max_len:int = None, position:bool = True, symbols:bool = False) -> np.array:
    if position:
        array = np.append(array, [0.0, 0.0, 0.0] * (max_len - array.shape[0]))
        array = array.reshape(int(array.shape[0]/3), 3)
    elif symbols:
        array = np.append(array, ['H'] * (max_len - array.shape[0]))
    else:
        array = np.append(array, [0] * (max_len - array.shape[0]))
    return array
