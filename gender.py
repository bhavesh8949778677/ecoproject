import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset from the CSV file
dataset = pd.read_csv('gender.csv')

# Preprocess the data
dataset['Name'] = dataset['Name'].str.lower()
dataset['Name'] = dataset['Name'].str.replace('[^a-z]', '')
dataset['Name_Length'] = dataset['Name'].apply(len)
dataset['Last_Letter'] = dataset['Name'].str[-1]

# Perform one-hot encoding on the last letter column
last_letter_encoded = pd.get_dummies(dataset['Last_Letter'], prefix='LastLetter')

# Concatenate the one-hot encoded columns with the dataset
dataset = pd.concat([dataset, last_letter_encoded], axis=1)

# Drop the original 'Last_Letter' column
dataset = dataset.drop('Last_Letter', axis=1)


X_train, X_test, y_train, y_test = train_test_split(dataset.drop(['Name', 'Gender'], axis=1), dataset['Gender'], test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train.values.reshape(-1, 1), y_train)
y_pred = model.predict(X_test.values.reshape(-1, 1))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)