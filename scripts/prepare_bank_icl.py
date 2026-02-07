
import pandas as pd
import os

# Create directory
os.makedirs("sampled_data/bank-marketing", exist_ok=True)

# Load data
df = pd.read_csv("data/bank-marketing.csv")

# Sample 20 rows
sampled = df.sample(n=20, random_state=42)

# Create sentence templates (matching generate.py logic manually for now to bootstrap)
sentences = []
for _, row in sampled.iterrows():
    # Template: A {{age}}-year-old {{marital}} client with {{education}} education is employed as {{job}}. ...
    marital = row['marital']
    education = row['education']
    job = row['job']
    default = "a" if row['default'] == 'yes' else "no"
    housing = "a" if row['housing'] == 'yes' else "no"
    loan = "a" if row['loan'] == 'yes' else "no"
    contact = row['contact']
    
    label_desc = "subscribed" if row['class-label'] == 'yes' else "not subscribed"
    
    s = f"A {row['age']}-year-old {marital} client with {education} education is employed as {job}. "
    s += f"The client has {default} credit default, balance of {row['balance']} euros, {housing} housing loan, and {loan} personal loan. "
    s += f"The client was contacted via {contact} on day {row['day']} of {row['month']}, with call duration {row['duration']} seconds. "
    s += f"The campaign involved {row['campaign']} contacts, with {row['pdays']} days since last contact, {row['previous']} previous contacts, and outcome {row['poutcome']}."
    s += f" The subscription decision is {label_desc}."
    
    sentences.append(s)

# Save
out_df = pd.DataFrame({'sentence': sentences, 'label': sampled['class-label'].map({'yes': 1, 'no': 0})})
out_df.to_csv("sampled_data/bank-marketing/sampled_sentences.csv", index=False)
print("Created sampled_data/bank-marketing/sampled_sentences.csv")
