import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data
training_data_df = pd.read_csv("sales_data_training.csv")

# Load test data
test_data_df = pd.read_csv("sales_data_test.csv")

# Scale to range 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)

# Print scaling info for later
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(
    scaler.scale_[8],
    scaler.min_[8]
))

# Save scaled files
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)

print("Scaled files created.")
