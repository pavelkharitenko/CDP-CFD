import torch, sys
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

dataset = DWDataset([
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-06-15-44-22-ndp-2-P600-forgiving-parameter-intermediate-savingsec-35000-ts.p",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-06-17-41-31-ndp-2-P600-wan-osprey-intermediate-savingsec-30000-ts.p",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-07-10-23-45-ndp-2-P600-low-dry-data-intermediate-savingsec-30000-ts.p",
    #r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-07-14-10-34-ndp-2-P600-mid-atomic-brass-intermediate-savingsec-35000-ts.p",
    r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-07-19-48-28-Dataset-NDP-2-P600-extremal-tempo-577.0sec-28851-ts.p"
    ])



x_train, x_val, y_train, y_val = train_test_split(dataset.x, dataset.y, train_size=0.75, test_size=0.25,
                                                      shuffle=True)

