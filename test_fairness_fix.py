
import numpy as np
from evaluate_fairness import demographic_parity, equalized_odds

# Test data with lists
y_pred = [0, 1, 0, 1, 1]
sensitive_attr = [0, 1, 0, 1, 0]
y_true = [0, 1, 0, 0, 1]

print("Testing demographic_parity with lists...")
try:
    res = demographic_parity(y_pred, sensitive_attr)
    print("✓ demographic_parity works:", res)
except Exception as e:
    print("✗ demographic_parity failed:", e)

print("\nTesting equalized_odds with lists...")
try:
    res = equalized_odds(y_true, y_pred, sensitive_attr)
    print("✓ equalized_odds works:", res)
except Exception as e:
    print("✗ equalized_odds failed:", e)
