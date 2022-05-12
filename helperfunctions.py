import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.init as init


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

def save_params(P,path):
    with open(path,'w') as f:
        f.write(json.dumps(P))

def load_params(path):
    with open(path,'r') as f:
        data = f.read()
    return json.loads(data)

def make_mask(model):
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    return mask

def prune_by_percentile(model,mask,percent):
    """
    Prunes the top percentile largest weights
    """
    step = 0
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        tensor = param.data.cpu().numpy()
        alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
        percentile_value = np.percentile(abs(alive), percent)
        # Convert Tensors to numpy and calculate
        weight_dev = param.device
        new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

        # Apply new weight and mask
        param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        mask[step] = new_mask
        step += 1

def re_init(model,mask):
    """
    Reinitializes the network given a mask
    """
    model.apply(weight_init)
    step = 0
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        weight_dev = param.device
        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
        step = step + 1

def og_init(model, mask, initial_state_dict):
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round((nonzero/total)*100,1))


def get_optimizer(model,solver_type,learning_rate,LBFGS_param=(10,4)):
    if solver_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif solver_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif solver_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif solver_type == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif solver_type == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), history_size=LBFGS_param[0], max_iter=LBFGS_param[1])
    else:
        optimizer = None 
    return optimizer

def get_scheduler(sch,optimizer,verbose):
    if sch==1:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=verbose, factor=0.5, eps=1e-12)
    elif sch == 2:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    return None

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)









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

