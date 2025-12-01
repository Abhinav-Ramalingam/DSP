import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic dataset with controlled overlaps
np.random.seed(42)
n = 300

# Create overlapping Age, Gender, Zip combinations
ages = np.random.choice(["20-29", "30-39", "40-49"], size=n)
genders = np.random.choice(["M", "F"], size=n)
zips = np.random.choice(["65*", "66*", "67*"], size=n)

# Create a sensitive attribute that correlates with QIDs
diagnoses = np.random.choice(["Diabetes", "Heart Disease", "Cancer"], size=n, p=[0.5, 0.3, 0.2])
outcomes = np.random.choice(["Recovered", "Deceased", "Critical"], size=n)

# Construct medical dataset
df_synth_medical = pd.DataFrame({
    "Age": ages,
    "Gender": genders,
    "Zip": zips,
    "Diagnosis": diagnoses,
    "Outcome": outcomes
})

# Attacker has same QIDs but no access to sensitive info
df_synth_demographic = df_synth_medical[["Age", "Gender", "Zip"]].drop_duplicates().sample(frac=0.7, random_state=1).reset_index(drop=True)

# Run k-anonymity analysis again
results = []
qid_columns = ["Age", "Gender", "Zip"]

for k in range(8, 15):  # Test for k = 2 to 10
    group_sizes = df_synth_medical.groupby(qid_columns).size().reset_index(name="group_size")
    df_kanon = df_synth_medical.merge(group_sizes, on=qid_columns)
    df_kanon.loc[df_kanon["group_size"] < k, ["Diagnosis", "Outcome"]] = "REDACTED"
    df_kanon = df_kanon.drop(columns=["group_size"])
    
    merged = pd.merge(df_kanon, df_synth_demographic, on=qid_columns)
    matches = merged[merged["Diagnosis"] != "REDACTED"].shape[0]
    
    results.append({"k": k, "matched_records": matches})

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["matched_records"], marker="o")
plt.title("Synthetic Data: Re-identification Risk vs k-Anonymity Level")
plt.xlabel("k value")
plt.ylabel("Number of Re-identification Matches")
plt.grid(True)
plt.tight_layout()
plt.show()

print(results_df)