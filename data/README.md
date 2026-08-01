 from src.lti_engine import robust_causal_direction

direction, confidence, method = robust_causal_direction(
    "Temperature", temp_series,
    "Electricity", elec_series,
    alpha=0.05
)
print(f"Direction: {direction} (confidence={confidence:.3f}, method={method})")
