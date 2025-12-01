import random
import numpy as np
import matplotlib.pyplot as plt
from sympy import nextprime

# Generate a 256-bit prime
def get_large_prime():
    return nextprime(2**25)

# Step 1: Generate k-degree polynomial (a0 is the secret, others are random)
def generate_polynomial(k, secret, prime):
    coeffs = [secret] + [random.randint(1, prime-1) for _ in range(k-1)]
    return coeffs

# Step 2: Compute (xi, yi) shares for n participants
def compute_shares(n, coeffs, prime):
    x_values = list(range(1, n+1))  # x-values: 1, 2, ..., n
    shares = [(x, sum(c * (x ** i) % prime for i, c in enumerate(coeffs)) % prime) for x in x_values]
    return shares

# Step 4: Select t random participants and their shares
def select_t_participants(shares, t):
    return random.sample(shares, t)

# Step 5: Lagrange Interpolation to reconstruct polynomial and extract secret
def lagrange_interpolation(shares, prime):
    def basis(j, x):
        num, den = 1, 1
        for m, (xm, _) in enumerate(shares):
            if m != j:
                num = (num * (x - xm)) % prime
                den = (den * (shares[j][0] - xm)) % prime
        return num * pow(den, -1, prime) % prime  # Modular inverse of denominator

    return sum(yj * basis(j, 0) for j, (_, yj) in enumerate(shares)) % prime  # Compute f(0)

# Step 6: Visualization of the polynomial and selected points
def plot_polynomial(coeffs, shares, selected_shares, prime):
    x_range = np.linspace(0, max(shares, key=lambda s: s[0])[0] + 2, 100)
    y_range = [(sum(c * (x ** i) % prime for i, c in enumerate(coeffs)) % prime) for x in x_range]

    plt.plot(x_range, y_range, label="Polynomial Curve")
    plt.scatter(*zip(*shares), color='blue', label="All Shares")
    plt.scatter(*zip(*selected_shares), color='red', label="Selected Shares", marker='x')

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Shamir Secret Sharing - Polynomial Reconstruction")
    plt.legend()
    plt.grid(True)
    plt.savefig("shamir_secret_sharing.png")

# Example Run
prime = get_large_prime()
k, n, t = 3, 6, 3  # Polynomial degree k-1, n total shares, t required to reconstruct
secret = 19111449  # The secret

coeffs = generate_polynomial(k, secret, prime)
shares = compute_shares(n, coeffs, prime)
selected_shares = select_t_participants(shares, t)
recovered_secret = lagrange_interpolation(selected_shares, prime)

# ** Step-by-Step Display **
print("\nStep 1: Generated Polynomial Coefficients")
print(f"Polynomial: f(x) = {' + '.join(f'{c}*x^{i}' for i, c in enumerate(coeffs))} (mod {prime})")

print("\nStep 2: All (xi, yi) Shares for Participants")
for i, (x, y) in enumerate(shares, 1):
    print(f"Participant {i}: (x={x}, y={y})")

print("\nStep 3: Selected t Participants for Reconstruction")
print("Selected Participants Table:")
print("ID | xi | yi")
print("---|----|----")
for i, (x, y) in enumerate(selected_shares, 1):
    print(f" {i} | {x} | {y}")

print("\nStep 4: Lagrange Interpolation - Recovered Secret")
print(f"Original Secret: {secret}")
print(f"Recovered Secret: {recovered_secret}")
assert secret == recovered_secret, "Secret reconstruction failed!"

# Plot Polynomial with Selected Points
plot_polynomial(coeffs, shares, selected_shares, prime)
