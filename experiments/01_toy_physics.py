import sys
import os

# هذا السطر ضروري لكي يتعرف الملف على مجلد src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lti_engine import robust_causal_direction

print("\n" + "="*57)
print("TEST: Toy Physics (Temperature vs Electricity)")
print("="*57)

# بيانات التجربة المذكورة في ورقتك البحثية
temp_series = [20, 21, 35, 38, 36, 22, 20]
elec_series = [100, 105, 140, 185, 195, 165, 130]

# تشغيل المحرك
direction, confidence, method = robust_causal_direction(
    "Temperature", temp_series,
    "Electricity", elec_series
)

print(f"\n  Causal Direction: {direction}")
print(f"  Confidence:       {confidence:.3f}")
print(f"  Method Used:      {method}\n")
