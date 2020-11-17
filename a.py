import numpy as np

dict_data = np.load('data/train_flow.npz')
data = dict_data['arr_0']
print(len(data))
