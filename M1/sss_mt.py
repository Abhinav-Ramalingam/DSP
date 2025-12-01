import random
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from mpyc.runtime import mpc

# Start MPyC to initialize field
await mpc.start()

# Use MPyC’s finite field (127-bit prime)
secfld = mpc.SecFld(2**127 - 1)
prime = secfld.field.order
print(f"Using MPyC finite field with prime = {prime}")

# Generate k-degree polynomial with secret as constant term
def generate_polynomial(k, secret, field):
    return [secret] + [random.randrange(1, field.field.order - 1) for _ in range(k - 1)]

# Evaluate polynomial at x using MPyC field
def eval_poly(x, coeffs, field):
    p = field.field.order
    result = 0
    for i, c in enumerate(coeffs):
        element = field.field(c)  # raw finite field element
        term = element * pow(x, i, p)
        result = (result + term.value) % p
    return result

# Compute shares (x, y)
def compute_shares(n, coeffs, field):
    return [(x, eval_poly(x, coeffs, field)) for x in range(1, n + 1)]

# Reconstruct secret using Lagrange interpolation
def lagrange_interpolation(shares, prime):
    def basis(j, x):
        num, den = 1, 1
        for m, (xm, _) in enumerate(shares):
            if m != j:
                num = (num * (x - xm)) % prime
                den = (den * (shares[j][0] - xm)) % prime
        return num * pow(den, -1, prime) % prime
    return sum(yj * basis(j, 0) for j, (_, yj) in enumerate(shares)) % prime

# Plotting the polynomial and selected shares
def plot_polynomial(coeffs, prime, selected_shares):
    x_vals = np.linspace(0, 10, 400)
    y_vals = [sum(c * (x ** i) for i, c in enumerate(coeffs)) % prime for x in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label="Polynomial Curve")

    selected_x = [share[0] for share in selected_shares]
    selected_y = [share[1] for share in selected_shares]
    plt.scatter(selected_x, selected_y, color='red', label='Selected Points', zorder=5)

    for i, (x, y) in enumerate(zip(selected_x, selected_y)):
        plt.text(x, y, f'({x},{y})', fontsize=9, ha='right')

    plt.title("Shamir Secret Sharing - Polynomial")
    plt.xlabel("x")
    plt.ylabel("y (mod p)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("polynomial_plot_with_selected_points.png")
    plt.show()

# Simulated Participant class (each runs in its own thread)
class Participant(threading.Thread):
    shared_pool = []
    lock = threading.Lock()
    ready_to_reconstruct = threading.Event()

    def __init__(self, participant_id, share, threshold, is_selected, peers):
        super().__init__()
        self.participant_id = participant_id
        self.share = share
        self.threshold = threshold
        self.is_selected = is_selected
        self.peers = peers

    def run(self):
        time.sleep(random.uniform(0.5, 2))  # Simulate delay

        with Participant.lock:
            if self.is_selected:
                for peer in self.peers:
                    print(f"Participant {self.participant_id} shares: {self.share} with {peer}")
                Participant.shared_pool.append(self.share)

            if len(Participant.shared_pool) >= self.threshold:
                Participant.ready_to_reconstruct.set()

        Participant.ready_to_reconstruct.wait()

        if len(Participant.shared_pool) >= self.threshold:
            with Participant.lock:
                secret = lagrange_interpolation(Participant.shared_pool[:self.threshold], prime)
                print(f"Participant {self.participant_id} reconstructed secret: {secret}")

# Dealer actions: generate secret, polynomial, and shares
def dealer_actions(field, k, n, t):
    secret = random.randint(1, field.field.order - 1)
    coeffs = generate_polynomial(k, secret, field)
    shares = compute_shares(n, coeffs, field)

    print("\n[Dealer] Secret:", secret)
    print("[Dealer] Polynomial Coefficients:", coeffs)
    print("[Dealer] Shares:")
    for s in shares:
        print(f"  {s}")

    selected_ids = random.sample(range(1, n + 1), t)
    selected_shares = [shares[i - 1] for i in selected_ids]
    print(f"\n[Dealer] Selected participants: {selected_ids}")

    plot_polynomial(coeffs, field.field.order, selected_shares)

    return selected_ids, selected_shares

# Main execution
k, n, t = 3, 6, 3  # Polynomial degree, total parties, threshold
selected_participants, selected_shares = dealer_actions(secfld, k, n, t)

participants = []
for i, share in enumerate(selected_shares):
    peers = [s for j, s in enumerate(selected_shares) if j != i]
    participants.append(Participant(selected_participants[i], share, t, True, peers))

for p in participants:
    p.start()
for p in participants:
    p.join()

await mpc.shutdown()