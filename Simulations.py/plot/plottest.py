import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y)

# Set major and minor locators
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Major ticks every 1 unit
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # 5 minor ticks between major ticks

ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  # Major ticks every 0.5 unit
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # 5 minor ticks between major ticks

# Turn on the grid for both major and minor ticks
ax.grid(which='both', linestyle='--', linewidth=0.5)

# Customize tick parameters for better visibility
ax.tick_params(axis='x', which='major', length=6, color='black')
ax.tick_params(axis='x', which='minor', length=4, color='gray')
ax.tick_params(axis='y', which='major', length=6, color='black')
ax.tick_params(axis='y', which='minor', length=4, color='gray')

# Show the plot
plt.show()
