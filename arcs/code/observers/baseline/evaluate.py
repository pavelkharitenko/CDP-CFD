import torch, sys
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DWPredictor
from dataset import DWDataset
import numpy as np

import matplotlib.pyplot as plt
from arcs.code.utils.utils import *

sys.path.append('../../../../../notify/')
from notify_script_end import notify_ending

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DWPredictor().to(device)
model.load_state_dict(torch.load(r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\2024-10-05-22-59-25-ndp-model-brilliant-vendor20000_eps.pth", weights_only=True))

model.eval()
#print(model(st))
plot_zy_yx_slices(model)
#plot_3D_forces(model)
