import sys
import os

# Allow script to read the engine from the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lti_engine import robust_causal_direction

print("\n" + "="*57)
print("TEST: Toy Physics (Temperature vs Electricity)")
print("="*57)

# Data from the research paper synthetic benchmark
temp_series = [20, 21, 35, 38, 36, 22, 20]
elec_series = [100, 105, 140, 185, 195, 165, 130]

# Run the engine
direction, confidence, method = robust_causal_direction(
    "Temperature", temp_series,
    "Electricity", elec_series
)

print(f"\n  Causal Direction: {direction}")
print(f"  Confidence Score: {confidence:.3f}")
print(f"  Method Used:      {method}\n")
