import torch, numpy as np

print(torch.cuda.is_available())

# YOLO v1:

# 1) split image in SxS grid
# each cell outputs a prediction with a corresponding BB, if BB's centerpoint is inside this cell, this cell is responsible
# to make sure only one BB per object, cell that contains object center is responsible for outputting BB
# find cell that contains object center

# 2) each BB output and label rel. to cell: [x_coord,y_coord,w_width,h_height] 
# (x and y of BB midpoint, in [0,1], but height and width may larger than 1)

# 