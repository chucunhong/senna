
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def test_plot():
    majority_roc_fpr = np.array([0.0, 0.2, 0.4, 1.0])
    majority_roc_tpr = np.array([0.0, 0.4, 0.8, 1.0])
    minority_roc_fpr = np.array([0.0, 0.3, 0.5, 1.0])
    minority_roc_tpr = np.array([0.0, 0.2, 0.6, 1.0])
    
    plt.figure()
    # Reproduce the call in utils.py lines 90-94
    # plt.fill(x, y, "y") -> "y" is color string?
    # Matplotlib fill signature: fill(*args, data=None, **kwargs)
    # args: x, y, [color]
    
    x = majority_roc_fpr.tolist() + np.flipud(minority_roc_fpr).tolist()
    y = majority_roc_tpr.tolist() + np.flipud(minority_roc_tpr).tolist()
    
    try:
        plt.fill(x, y, "y")
        print("Plotting successful")
    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plot()
