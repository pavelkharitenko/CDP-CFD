import torch, sys, pickle
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DWPredictor
from dataset import DWDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
sys.path.append('../../uav/')
sys.path.append('../../utils/')

from uav import *
from utils import *


editpath = r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-07-14-10-34-ndp-2-P600-mid-atomic-brass-intermediate-savingsec-35000-ts.p"
experiment = None
with open(editpath, 'rb') as handle:
    experiment = pickle.load(handle)
    #print("Loaded " + experiment["exp_name"])
    experiment['bias'] = 29.77
    experiment['exp_name'] = "2024-10-07-14-10-34-ndp-2-P600-mid-atomic-brass-intermediate-savingsec-35000-ts"

    print(experiment)



with open("2024-10-07-14-10-34-ndp-2-P600-mid-atomic-brass-intermediate-savingsec-35000-ts.p", 'wb') as handle:
    pickle.dump(experiment, handle)