import numpy as np
import matplotlib.pyplot as plt

# Define the polynomial function
def polynomial(x):
    return (
        128195624231432157091713560391881993388 +
        79271169439095337974508381727176319419 * x +
        122246645436092756414141582736448092356 * x**2
    )

# Generate x values within a reasonable range
x_values = np.linspace(-5, 5, 400)
y_values = polynomial(x_values)

# Given points to mark
points = [
    (1, 159572255646151019748676221139622299436),
    (3, 105099473789799124889014520474370937033),
    (4, 19250060518728367372390159061379268582)
]

# Plot the polynomial
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label=r'$y = 128195624231432157091713560391881993388 + 79271169439095337974508381727176319419x + 122246645436092756414141582736448092356x^2$', color='blue')

# Mark the specific points
for x, y in points:
    plt.scatter(x, y, color='red', zorder=3)
    plt.text(x, y, f'({x}, {y})', fontsize=10, verticalalignment='bottom', horizontalalignment='right')

# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolated Polynomial for Secret Sharing')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)

# Show plot
plt.show()
