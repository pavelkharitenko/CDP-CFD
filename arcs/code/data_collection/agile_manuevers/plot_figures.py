import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utils/')
from utils import *

#create_demo_figure_1(find_file_with_substring("demo_1_flybelow"))

#create_demo_figure_1(find_file_with_substring("short_demo_1_flyabove"))

create_demo_subfigure_1(find_file_with_substring("demo_1_flybelow"), find_file_with_substring("demo_1_flyabove"))

plt.show()