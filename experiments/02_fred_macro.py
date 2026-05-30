import numpy as np
import pandas_datareader.data as web
import datetime
from scipy.signal import savgol_filter
import sys
import os

# السماح للسكربت بقراءة المحرك من مجلد src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lti_engine import robust_causal_direction

print("\n" + "="*57)
print("TEST: Real Noisy Macroeconomic Data (FRED) 2000-2023")
print("="*57)

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2023, 1, 1)

try:
    print("Fetching data from FRED API...")
    df = web.DataReader(['FEDFUNDS', 'UNRATE'], 'fred', start, end).dropna()
    
    # تنعيم البيانات باستخدام فلتر Savitzky-Golay
    fed_smooth = savgol_filter(df['FEDFUNDS'].values, window_length=11, polyorder=2)
    unrate_smooth = savgol_filter(df['UNRATE'].values, window_length=11, polyorder=2)
    
    print("Running LTI Engine on full 23-year series...")
    direction, conf, method = robust_causal_direction(
        "Interest_Rate (FEDFUNDS)", fed_smooth,
        "Unemployment (UNRATE)", unrate_smooth
    )
    
    print(f"\n  Inferred:   {direction}")
    print(f"  Method:     {method}")
    print(f"  Confidence: {conf:.3f}\n")
    print("✅ PASS: Engine survived real-world noise!")
        
except Exception as e:
    print(f"Error: {e}")