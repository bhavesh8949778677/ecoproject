import random
import pandas as pd

# List of example names
names = ['John', 'Robert', 'Michael', 'William', 'David', 'Mary', 'Jennifer', 'Linda', 'Patricia', 'Susan']

# Generating a random dataset
dataset = pd.DataFrame(columns=['Name', 'Gender'])

for name in names:
    # Generate a random gender label
    gender = random.choice(['Male', 'Female'])
    
    # Add name and gender to the dataset
    dataset = dataset._append({'Name': name, 'Gender': gender}, ignore_index=True)

# Saving the dataset to a CSV file
dataset.to_csv('gender.csv', index=False)