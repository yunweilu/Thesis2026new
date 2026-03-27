import numpy as np
import matplotlib.pyplot as plt

# --- Generate a smooth fluctuating curve (black) ---
rng = np.random.default_rng(7)
t = np.linspace(0, 10, 50)

noise = rng.normal(-1, 1, size=t.size)
# Normalize noise to [-1, 1]
noise = 2 * (noise - noise.min()) / (noise.max() - noise.min()) - 1
k = np.linspace(-3, 3, 151)
kernel = np.exp(-0.5 * k**2)
kernel /= kernel.sum()
f = 0.1*np.cos( 0.5+noise ) 

# --- Blue curve with opposite derivative ---
black_baseline = 0.85
blue_baseline  = 0.35
black = black_baseline + f
blue  = blue_baseline  - f

# Constant total energy (solid blue line)
total_const = np.mean(black + blue) * np.ones_like(t)


# --- Plot with two subplots ---
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(5, 5), dpi=160)
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)


# Main plot (top)
ax1 = fig.add_subplot(gs[0])
ax1.plot(t, black, color="black", linewidth=2.6, solid_capstyle="round")
ax1.plot(t, blue,  color="blue",  linewidth=2.6, solid_capstyle="round")
ax1.plot(t, total_const, color="blue", linewidth=1.6)

# Show y-axis, hide x-axis ticks
ax1.set_ylabel("")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.tick_params(bottom=False, left=True)
for side in ["bottom", "top", "right"]:
    ax1.spines[side].set_visible(False)
ax1.spines["left"].set_visible(True)
ax1.set_xlim(t.min(), t.max())
ymin = min(black.min(), blue.min(), total_const.min())
ymax = max(black.max(), blue.max(), total_const.max())
pad = 0.08 * (ymax - ymin)
ax1.set_ylim(ymin - pad, ymax + pad)
x0, x1 = ax1.get_xlim()
y0, y1 = ax1.get_ylim()
# y-axis arrow (main plot)
ax1.annotate(
    "", xy=(x0, y1), xytext=(x0, y0),
    arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color="black", shrinkA=0, shrinkB=0),
    clip_on=False
)

# Noise plot (bottom)
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(t, noise, color="#888888")
ax2.set_xlabel("t")
ax2.set_ylabel("")
ax2.set_yticks([])
ax2.tick_params(bottom=True, left=True)
for side in ["top", "right"]:
    ax2.spines[side].set_visible(False)
for side in ["bottom", "left"]:
    ax2.spines[side].set_visible(True)
ax2.set_xlim(t.min(), t.max())
# x-axis arrow (noise plot)
x0, x1 = ax2.get_xlim()
y0, y1 = ax2.get_ylim()
ax2.annotate(
    "", xy=(x1, y0), xytext=(x0, y0),
    arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color="black", shrinkA=0, shrinkB=0),
    clip_on=False
)
# y-axis arrow (noise plot)
ax2.annotate(
    "", xy=(x0, y1), xytext=(x0, y0),
    arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color="black", shrinkA=0, shrinkB=0),
    clip_on=False
)

plt.tight_layout()
plt.savefig('illustration.svg', bbox_inches='tight')
plt.show()
