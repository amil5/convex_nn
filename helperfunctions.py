import numpy as np
import torch 
from torch.utils.data import Dataset

def print_params(P,general=True,ncvx=False,cvx=False):
    pl = []
    for k,v in P.items():
        start = k.split('_')[0]
        if (start == 'cvx' and cvx) or (start == 'ncvx' and ncvx) or \
           (start != 'cvx' and start != 'ncvx' and general):
           pl.append((k,v))
    param_len = len(max(pl,key=lambda x: len(x[0]))[0]) + 1
    print("{1:<{0}}: Value\n{2:}".format(param_len,"Parameter","="*(param_len+7)))
    print("\n".join(["{1:<{0}}: {2:}".format(param_len,k,v) for k,v in pl]))

def check_if_already_exists(element_list, element):
    # check if element exists in element_list where element is a numpy array
    for i in range(len(element_list)):
        if np.array_equal(element_list[i], element):
            return True
    return False

class PrepareData(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X) if not torch.is_tensor(X) else X
        self.y = torch.from_numpy(y) if not torch.is_tensor(y) else y
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PrepareData3D(Dataset):
    def __init__(self, X, y, z):
        self.X = torch.from_numpy(X) if not torch.is_tensor(X) else X
        self.y = torch.from_numpy(y) if not torch.is_tensor(y) else y
        self.z = torch.from_numpy(z) if not torch.is_tensor(z) else z

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.z[idx]

def one_hot(labels, num_classes=10):
    y = torch.eye(num_classes) 
    return y[labels.long()] 

