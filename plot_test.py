import numpy as np
import matplotlib.pyplot as plt


BOX_LINEWIDTH = 2
GRID_LINEWIDTH = 1
TICK_WIDTH = 2



t = np.arange(1000)
#x = 20*np.log10(100/(100+t))
x = (t/(50+t))*(1+np.sin(2*np.pi*0.01*t))
error = np.cumsum(np.random.choice((-0.05,0.05),1000))
x2 = 0.87*(t/(70+t))*(1+np.sin(2*np.pi*0.01*t+error))

plt.plot(t,x, linewidth=2, color="gray", label="a")
plt.plot(t,x2, linewidth=2, color="k", label="b")

plt.title("V$_{DS}$ vs t", fontweight="bold", size=14)
plt.xlabel("time", fontweight="bold", size=12)
plt.ylabel(r"V$_{DS}$(t)", fontweight="bold", size=12)


font = {'weight' : 'bold',}
plt.rc('font', **font)
ax = plt.gca()
xax = ax.get_xaxis()
yax = ax.get_yaxis()
xax.set_ticks_position("both")
yax.set_ticks_position("both")
ax.tick_params(axis="both",which="both", direction="in", width=TICK_WIDTH)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(BOX_LINEWIDTH)
    plt.grid(linewidth=GRID_LINEWIDTH, color="dimgray", linestyle=":")

# fix label/titles
plt.title(ax.get_title(), fontweight="bold", size=14)
plt.xlabel(ax.get_xlabel(), fontweight="bold", size=12)
plt.ylabel(ax.get_ylabel(), fontweight="bold", size=12)
fontProperties = {'family':'sans-serif',
    'weight' : 'bold'}
ax.set_xticklabels(ax.get_xticks(), fontProperties)
ax.set_yticklabels(ax.get_yticks(), fontProperties)
plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=False, title=None, fancybox=False,
           frameon=False)
plt.show()
