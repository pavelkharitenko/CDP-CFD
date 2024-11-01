import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def median_filter(data, kernel_size):
    return medfilt(data, kernel_size=kernel_size)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def exponential_moving_average(data, alpha):
    ema = [data[0]]
    for x in data[1:]:
        ema.append(alpha * x + (1 - alpha) * ema[-1])
    return ema

def low_pass_filter(data, alpha):
    filtered_data = [data[0]]
    for x in data[1:]:
        filtered_data.append(filtered_data[-1] + alpha * (x - filtered_data[-1]))
    return filtered_data


data = np.load('throttle_thrust_map_005_80_500Hz.npz')
throttle_list = data['throttle_list']
thrust_list = data['thrust_list']


fig = plt.subplot()
fig.plot(thrust_list[:5200], throttle_list[:5200], label="raw measurements")

# use filters:

fig = plt.subplot()
#fig.plot(throttle_list, median_filter(thrust_list, 3), label="median filter, ks=3")
#fig.plot(median_filter(thrust_list, 21), label="median filter, ks=21")
#fig.plot(moving_average(thrust_list, 50), label="moving avg filter, ks=50")
#fig.plot(exponential_moving_average(thrust_list, 0.05), label="exp. moving avg filter, ks=0.1")
#fig.plot(low_pass_filter(thrust_list, 0.045), label=" low pass filter, ks=0.045")
#fig.plot(moving_average(thrust_list, 30), label="moving avg filter, ks=30")
fig.plot(moving_average(low_pass_filter(thrust_list, 0.045),7)[:5200], throttle_list[:5200], label=" low pass filter 0.045 + avg. 10")


fig.set_title("throttle vs force")
fig.set_xlabel("throttle input value")
fig.set_ylabel("Raw measurement on JFT sensor (N)")

plt.legend()
plt.show()



from scipy.interpolate import UnivariateSpline

# Sample data points
x = throttle_list[300:5200]
y = moving_average(low_pass_filter(thrust_list, 0.045),7)[300:5200]

# Spline fit with smoothing factor
spline = UnivariateSpline(x, y)
spline.set_smoothing_factor(1.5)  # Adjust for more or less smoothing

# Generate smooth values
x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = spline(x_smooth) # thrusts

# Plot
plt.scatter(x, y, color='red')
plt.plot(throttle_list[:5200], thrust_list[:5200], label="raw measurements")
plt.plot(x_smooth, y_smooth, color='blue')

plt.show()



def find_nearest(array, value):
    ''' Find nearest value is an array '''
    idx = (np.abs(array-value)).argmin()
    return idx


#plt.plot(thrust_list[:5200], throttle_list[:5200], label="raw measurements")
#plt.plot([y_smooth[find_nearest(y_smooth, x)] for x in x_smooth])
#plt.show()





#fig.plot(median_filter(thrust_list, 31), label="median avg filter, ks=31")

#fig.plot(moving_average(thrust_list, 50), label="moving avg filter, ks=60")

#fig.plot(moving_average(thrust_list, 60), label="moving avg filter, ks=60")







