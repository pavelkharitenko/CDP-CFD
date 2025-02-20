import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utils/')
from utils import *

#create_demo_figure_1(find_file_with_substring("demo_1_flybelow"))

#create_demo_figure_1(find_file_with_substring("short_demo_1_flyabove"))

# figure 2 plot
#create_demo_subfigure_1(find_file_with_substring("demo_1_flybelow"), find_file_with_substring("demo_1_flyabove"))

create_demo_subfigure_1(find_file_with_substring("3_swapping_demo_0753it_200Hz_80_005_len4070ts_5_iterations"),
                        find_file_with_substring("3_swapping_demo_3004it_200Hz_80_005_len1754ts_6_iterations"),
                       #find_file_with_substring("3_swapping_demo_1505it_200Hz_80_005_len3740ts_7_iterations.npz") 
                        )
plt.show()

# figure scenario exp plot
#create_scenario_exp_figure(find_file_with_substring("1_flybelow_exp_baseline"))

#create_multiple_scenarios_figure([find_file_with_substring("1_flybelow_exp_agile_200Hz_80_005_len2199ts_baseline_175"),
#                                  find_file_with_substring("1_flybelow_exp_agile_200Hz_80_005_len2199ts_agile_sn_175"),
#                                  find_file_with_substring("1_flybelow_exp_agile_200Hz_80_005_len2199ts_agile_sn_dot05")],
#                                  start_seconds_list=[8.0, 8.0,8.0], end_seconds_list=[3.0, 3.0,3.0]
#                                )

plt.show()