import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# generate some example data
x = range(100)
y = [x**2 for x in range(100)]

# create the main plot and the zoomed-in plot as subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# plot the main data on the left subplot
ax1.plot(x, y, 'b-')

# set the limits of the left subplot to show only the range from 0 to 100
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 10000)

# create the zoomed-in plot on the right subplot
axins = zoomed_inset_axes(ax1, zoom=4, loc='upper right')

# plot the same data on the zoomed-in plot
axins.plot(x, y, 'b-')

# set the limits of the zoomed-in plot to show only the range from 0 to 20
axins.set_xlim(0, 20)
axins.set_ylim(0, 400)

# add a rectangle to the main plot to indicate the area of the zoomed-in plot
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# add labels and titles to the subplots
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Main Plot')

axins.set_xlabel('X')
axins.set_ylabel('Y')
axins.set_title('Zoomed-In Plot')

plt.savefig("annotation_patterns/test.pdf")
