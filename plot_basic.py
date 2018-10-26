from matplotlib import pyplot as plt
import numpy as np

lw_size = 4.5
marker_size = 10
legend_font = 24
label_font = 28
tick_font = 22
major_tick_size = 1.5
minor_tick_size = 1.3

# simple plot
fig = plt.figure(figsize=(14, 10))
axes = fig.add_axes([0, 0, 1, 1])

x_data = np.linspace(20, 40)
y_data = np.linspace(20, 4000)
axes.plot(x_data, y_data, 'k--', linewidth=lw_size, markersize=marker_size, label='1')

axes.set_yscale('log')
axes.legend(fontsize=legend_font, loc='upper left')

for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_font)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_font)

axes.set_xlabel(r'$x$', fontsize=label_font)
axes.set_ylabel(r'$y$', fontsize=label_font)

# fig.savefig('test1.eps', bbox_inches='tight')


# two subplots
x_data = np.logspace(1, 4)
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x_data, y_data, 'k--', linewidth=lw_size, label='1')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x_data, y_data, 'k--', linewidth=lw_size, label='2')

for ax in [ax1, ax2]:
    ax.set_xlim(left=0.0, right=10**4)
    ax.set_xlabel(r'$t$, s', fontsize=20)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 3))
    ax.legend(fontsize=legend_font, loc='center right')

# fig.savefig('test2.eps', bbox_inches='tight')
