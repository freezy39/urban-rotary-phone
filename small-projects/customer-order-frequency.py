import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file (update the path if needed)
df = pd.read_csv("C:/Users/jash.farrell/Downloads/orders-2025-08-26.csv")

# Convert order date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows with missing customer ID or order ID
df = df.dropna(subset=['Customer ID', 'Order ID'])

# Group by Customer ID and count orders (frequency)
frequency = df.groupby('Customer ID')['Order ID'].nunique().reset_index()
frequency.columns = ['Customer ID', 'Frequency']

# Calculate mean and standard deviation
mean_freq = frequency['Frequency'].mean()
std_freq = frequency['Frequency'].std()

# Plot histogram
plt.figure(figsize=(12, 6))
sns.histplot(frequency['Frequency'], bins=50, kde=False, color='skyblue')

# Add standard deviation lines
for i in range(-3, 4):
    plt.axvline(mean_freq + i * std_freq, color='red' if i == 0 else 'gray', linestyle='--')
    if i != 0:
        plt.text(mean_freq + i * std_freq, plt.ylim()[1]*0.9, f'{i:+}σ', color='black', ha='center')

plt.title("Customer Frequency Distribution with Std Dev Lines")
plt.xlabel("Number of Orders (Frequency)")
plt.ylabel("Number of Customers")
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Save cleaned frequency data
frequency.to_csv("customer_frequency_summary.csv", index=False)
