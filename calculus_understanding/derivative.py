import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**2

def f_prime(x):
    return 2 * x

# Define x range for plot
x = np.linspace(-2, 4, 400)
y = f(x)

# Choose point to analyze derivative
x0 = 1
y0 = f(x0)

# h values: decreasing steps to show secant approaching tangent
h_values = [2, 1, 0.5, 0.1, 0.01]

# Plot the function
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='f(x) = x²', color='blue', linewidth=2)

# Draw multiple secants
colors = ['red', 'orange', 'green', 'purple', 'gray']
for i, h in enumerate(h_values):
    x1 = x0 + h
    y1 = f(x1)
    slope_secant = (y1 - y0) / h
    secant_line = slope_secant * (x - x0) + y0
    label = f'Secant h={h}'
    plt.plot(x, secant_line, linestyle='--', linewidth=1.8, color=colors[i], label=label)

# Draw exact tangent line
tangent_slope = f_prime(x0)
tangent_line = tangent_slope * (x - x0) + y0
plt.plot(x, tangent_line, linestyle='-', color='black', linewidth=2.5, label='Tangent (h→0)')

# Mark the fixed point
plt.scatter([x0], [y0], color='black')
plt.text(x0, y0 + 0.5, f"A ({x0}, {y0})", fontsize=12)

# Final plot styling
plt.title('Derivative of f(x) = x² using Secant → Tangent (h → 0)', fontsize=14)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.legend()
plt.grid(True)
plt.show()
