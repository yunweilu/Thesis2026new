import numpy as np
import matplotlib.pyplot as plt

# Parameters
duration = 1.0      # seconds
fs = 100           # sampling rate, Hz
N = int(duration * fs)  # total number of samples

# Generate white noise
white_noise = np.random.normal(0, 1, N)

# Time array for plotting
t = np.linspace(0, duration, N, endpoint=False)

# Plot the white noise curve only, no axes or labels
fig, ax = plt.subplots()
# Use a grey color similar to the label in your image
ax.plot(t, white_noise, color='gray', linewidth=4)
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('white_noise_curve.svg', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()