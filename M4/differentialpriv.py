import numpy as np
import matplotlib.pyplot as plt

# Dataset
votes = [
    {"name": "John", "gender": "male", "vote": "Bob"},
    {"name": "Mike", "gender": "male", "vote": "Bob"},
    {"name": "Mikaela", "gender": "female", "vote": "Alice"},
    {"name": "Anna", "gender": "female", "vote": "Alice"},
    {"name": "Daniela", "gender": "female", "vote": "Alice"}
]


true_count = sum(1 for p in votes if p["vote"] == "Alice" and p["gender"] == "female")


epsilon = 1.0
sensitivity = 1
scale = sensitivity / epsilon

num_queries = 100

epsilons = np.linspace(0.01, 2.0, num_queries)  
noisy_outputs = []

for diff_ep in epsilons:
    new_scale = sensitivity / diff_ep
    noise = np.random.laplace(loc=0.0, scale=new_scale)
    noisy_result = true_count + noise
    noisy_outputs.append(noisy_result)

# Plot results
plt.plot(epsilons, noisy_outputs, marker='o', linestyle='-', alpha=0.7)
plt.axhline(y=true_count, color='red', linestyle='--', label='True Count')
plt.xlabel("Epsilon")
plt.ylabel("Noisy Output")
plt.title("Effect of Varying Epsilon on Laplace Noise")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

num_queries = 100
noisy_counts = []

for _ in range(num_queries):
    noise = np.random.laplace(loc=0.0, scale=scale)
    noisy_count = true_count + noise
    noisy_counts.append(noisy_count)


estimated_count = np.mean(noisy_counts)


print(f"True count: {true_count}")
print(f"Estimated count after {num_queries} queries: {estimated_count:.2f}")

estimated_count = np.mean(noisy_counts)

# Print result
print(f"True count: {true_count}")
print(f"Estimated count after {num_queries} queries: {estimated_count:.2f}")

# Plot the noisy results
plt.hist(noisy_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black', label="Noisy Counts")
plt.axvline(true_count, color='red', linestyle='--', linewidth=2, label="True Count")
plt.axvline(estimated_count, color='green', linestyle='-', linewidth=2, label="Estimated Mean")
plt.title("Composition Attack on Laplace Mechanism")
plt.xlabel("Noisy Count of Female Votes for Alice")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
