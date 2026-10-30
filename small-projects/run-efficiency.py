import pandas as pd
import numpy as np
import time

# Create dummy data with 10 million rows
rows = 1_000_000_000
df = pd.DataFrame({'value': np.random.randint(1, 100, size=rows)})

# Define chunking and processing logic
def process_chunks(df, split_count):
    chunk_size = len(df) // split_count
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size].copy() for i in range(split_count)]
    
    start = time.time()
    for chunk in chunks:
        chunk['value'] = chunk['value'] * 2  # Simulated processing
    end = time.time()
    
    return split_count, round(end - start, 2)

# Test with different chunk counts
for splits in [1, 2, 5, 10, 20]:
    count, duration = process_chunks(df, splits)
    print(f"{count} chunk(s): {duration} seconds")
