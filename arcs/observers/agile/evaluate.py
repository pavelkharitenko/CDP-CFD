import torch, sys
from model import DWPredictor
import matplotlib.pyplot as plt
#from notify_script_end import notify_ending
sys.path.append('../../../../../notify/')
sys.path.append('../../utils/')
from utils import *


evaluate_multiple = True
device = "cuda" if torch.cuda.is_available() else "cpu"

if evaluate_multiple:
    model_no_y = DWPredictor().to(device)
    model_no_y.load_state_dict(torch.load(r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-12-25-27-NDP-Li-Model-sn_scale-None-144k-datapoints-corrected-bias-complicated-tropics20000_eps.pth", weights_only=True))

    model_y_2 = DWPredictor().to(device)
    model_y_2.load_state_dict(torch.load(r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-10-10-56-56-NDP-Li-Model-sn_scale-2-144k-datapoints-corrected-bias-edible-status20000_eps.pth", weights_only=True))

    model_y_4 = DWPredictor().to(device)
    model_y_4.load_state_dict(torch.load(r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\observers\ndp\2024-10-09-21-24-10-NDP-Li-Model-sn_scale-4-144k-datapoints-corrected-bias-sizzling-speed20000_eps.pth", weights_only=True))

    plot_model_compare([model_y_2, model_y_4, model_no_y])

else:
    model = DWPredictor().to(device)
    model.load_state_dict(torch.load(r"2024-10-08-11-56-55-NDP-Li-Model-sn_scale-6-small-outlet20000_eps.pth", weights_only=True))
    plot_zy_yx_slices(model)
    plot_z_slices(model)
    plot_3D_forces(model)
exit(0)
