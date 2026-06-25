
import pandas as pd
import os

data_dir = "/Users/doanthang/Desktop/Code/Fairness/Dist_Learn_Fair_Tab/data"
datasets = ["german-credit-data.csv", "credit-scoring.csv", "adult.csv"]

print(f"{'Dataset':<25} {'Samples':<10} {'Attr':<5} {'PosLabel':<15} {'PosCount':<10} {'NegCount':<10} {'IR (+:-)':<10}")

for filename in datasets:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        continue
    
    try:
        df = pd.read_csv(path)
        
        # Target is class-label
        if "class-label" not in df.columns:
            print(f"{filename:<25} Error: class-label not found")
            continue

        n_samples = len(df)
        n_attrs = len(df.columns) # Total columns as per User's apparent convention (Adult=15)
        
        counts = df["class-label"].value_counts()
        
        # Determine Positive Label logic based on User Table
        # Adult: >50K (Minority)
        # Bank: yes (Minority)
        # Credit: 1 (?)
        # German: Good/Bad ? (Usually Good is Majority)
        # Dutch: 1 (?)
        
        pos_label = None
        
        if filename == "adult.csv":
            # Check for >50K or similar
            keys = counts.keys()
            for k in keys:
                if ">50K" in str(k):
                    pos_label = k
                    break
        elif filename == "german-credit-data.csv":
            # Usually 'good' is Pos in fairness (privileged outcome), but minority class is often target.
            # German Good=700, Bad=300.
            # If target is Risk(Bad), then Pos=Bad.
            # Table says: German Credit ... Target is credit risk (Good/Bad).
            # Usually in IR notations: Majority:Minority or 1:Ratio.
            # If User used 1:3.03 for Adult (Pos is >50K, ~25%), then 1 (Pos) : 3 (Neg).
            # So Ratio is Neg/Pos.
            # Let's print raw counts to decide.
            pos_label = "good" if "good" in counts else 1
        elif filename == "credit-scoring.csv":
            # User said "class-label (1)"
            pos_label = 1
        if pos_label is not None and pos_label in counts:
            p_count = counts[pos_label]
            n_count = n_samples - p_count
            if p_count > 0:
                ratio = n_count / p_count
                ir_str = f"1:{ratio:.2f}"
            else:
                ir_str = "N/A"
        else:
            # Fallback for display
            keys = list(counts.keys())
            if len(keys) == 2:
                # Assume minority is positive for calculation
                c1, c2 = counts[keys[0]], counts[keys[1]]
                if c1 < c2:
                    p_count = c1
                    pos_label = keys[0]
                    n_count = c2
                else:
                    p_count = c2
                    pos_label = keys[1]
                    n_count = c1
                ratio = n_count / p_count
                ir_str = f"1:{ratio:.2f} (Assume {pos_label} is +)"
            else:
                pos_label = "N/A"
                p_count = 0
                n_count = 0
                ir_str = "N/A"

        print(f"{filename:<25} {n_samples:<10} {n_attrs:<5} {str(pos_label):<15} {p_count:<10} {n_count:<10} {ir_str:<10}")

    except Exception as e:
        print(f"{filename}: {e}")
