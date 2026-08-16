"""Quick test for the MIT-BIH Kafka producer."""
from kafka_mitbih_producer import iter_raw_signal_chunks

count = 0
for chunk in iter_raw_signal_chunks("202"):
    count += 1
    if count <= 3:
        print(f"Chunk {count}: sample={chunk['sample_index']}, r_peaks={chunk['r_peaks']}, samples={len(chunk['samples'])}, final={chunk['is_final']}")
    if count > 5:
        break
print(f"Total chunks in first 5: {count}")
print("MIT-BIH producer test PASSED")