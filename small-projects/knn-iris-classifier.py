"""
CSC525 Module 2 CT
KNN Iris Classifier

This program implements a K-Nearest Neighbors (KNN) classifier from scratch.
It predicts the iris species based on four measurements:
sepal length, sepal width, petal length, and petal width (in centimeters).
"""

import csv
import math
import os
from collections import Counter


def load_iris_data(csv_path):
    """Load iris data from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = []

    with open(csv_path, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Skip header row
    for row in rows[1:]:
        if len(row) < 5:
            continue

        sepal_length = float(row[0])
        sepal_width = float(row[1])
        petal_length = float(row[2])
        petal_width = float(row[3])
        label = row[4]

        data.append((sepal_length, sepal_width, petal_length, petal_width, label))

    return data


def euclidean_distance(a, b):
    """Calculate Euclidean distance between two 4D points."""
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2 +
        (a[3] - b[3]) ** 2
    )


def knn_predict(training_data, query_point, k):
    """Predict iris species using KNN."""
    distances = []

    for row in training_data:
        features = row[:4]
        label = row[4]
        dist = euclidean_distance(features, query_point)
        distances.append((dist, label))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    votes = Counter(label for _, label in neighbors)
    return votes.most_common(1)[0][0]


def get_user_input():
    """Prompt user for iris measurements."""
    print("Enter iris measurements in centimeters.")
    sepal_length = float(input("Sepal length: "))
    sepal_width = float(input("Sepal width: "))
    petal_length = float(input("Petal length: "))
    petal_width = float(input("Petal width: "))
    return (sepal_length, sepal_width, petal_length, petal_width)


def main():
    print("KNN Iris Classifier")
    print("----------------------------")

    default_csv = "iris.csv"
    default_k = 5

    csv_path = input(f"Enter CSV path (press Enter for {default_csv}): ").strip()
    if not csv_path:
        csv_path = default_csv

    k_input = input(f"Enter K value (press Enter for {default_k}): ").strip()
    k = default_k if not k_input else int(k_input)

    if k % 2 == 0:
        print("Note: Even K values can cause ties.")

    try:
        training_data = load_iris_data(csv_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded {len(training_data)} records.")
    print("----------------------------")

    try:
        query = get_user_input()
        prediction = knn_predict(training_data, query, k)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    print("----------------------------")
    print(f"Predicted iris species: {prediction}")
    print("Done.")


if __name__ == "__main__":
    main()
