# cell 1

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.optim as optim
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)  # Adjust num_labels as needed

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load data
train_data = pd.read_csv('train_scaled.csv', delimiter=',', nrows=200)
valid_data = pd.read_csv('valid_scaled.csv', delimiter=',', nrows=200)
test_data = pd.read_csv('test_scaled.csv', delimiter=',', nrows=200)

# Calculate class weights
unique_classes = np.unique(train_data['label'])
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_data['label'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Combine code and code_comment columns into a single input
train_data['combined_input'] = train_data['code'] + ' ' + train_data['code_comment']
valid_data['combined_input'] = valid_data['code'] + ' ' + valid_data['code_comment']
test_data['combined_input'] = test_data['code'] + ' ' + test_data['code_comment']

# Ensure combined_input is treated as a string and fill NaN values
train_data['combined_input'] = train_data['combined_input'].fillna('').astype(str)
valid_data['combined_input'] = valid_data['combined_input'].fillna('').astype(str)
test_data['combined_input'] = test_data['combined_input'].fillna('').astype(str)

# Tokenize data
def tokenize_function(data):
    return tokenizer(data['combined_input'].tolist(), padding='max_length', truncation=True, max_length=512, return_tensors="pt")

train_encodings = tokenize_function(train_data)
valid_encodings = tokenize_function(valid_data)
test_encodings = tokenize_function(test_data)

# Convert to PyTorch Dataset
class BugDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BugDataset(train_encodings, train_data['label'].values)
valid_dataset = BugDataset(valid_encodings, valid_data['label'].values)
test_dataset = BugDataset(test_encodings, test_data['label'].values)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# Set up optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
# Define the weighted loss function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

# Training loop
model.train()
for epoch in range(3):  # Number of epochs
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        # Move batch to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in valid_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy}")

# cell 2

print("Data type of labels in train_data:",train_data['label'].dtype)
print("Unique values in train_data['label']:",train_data['label'].unique())
print("Data type of labels in valid_data:", valid_data['label'].dtype)
print("Unique values in valid_data['label']:", valid_data['label'].unique())
print("Data type of labels in test_data:", test_data['label'].dtype)
print("Unique values in test_data['label']:", test_data['label'].unique())

# cell 3

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Evaluate the model on the test set
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for batch in DataLoader(test_dataset, batch_size=8, shuffle=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print("Test Set Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print a few examples of predicted and actual labels
print("Sample Predictions:")
for i in range(5):  # Display the first 5 predictions
    print(f"Predicted: {all_predictions[i]}, Actual: {all_labels[i]}")

# cell 4

import seaborn as sns

# Check the distribution of classes
# sns.countplot(x='label', data=valid_data)
# plt.title('Class Distribution in validing Data')
# plt.show()

# Print the number of values in each category
category_counts = train_data['label'].value_counts()
print("Number of values in each category:")
print(category_counts)
category_counts = valid_data['label'].value_counts()
print("Number of values in each category:")
print(category_counts)
category_counts = test_data['label'].value_counts()
print("Number of values in each category:")
print(category_counts)

# cell 5

# Generate a classification report
report = classification_report(all_labels, all_predictions)
print("Classification Report:")
print(report)

# cell 6

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# cell 7

# Save the model
model.save_pretrained('./roberta_bug_severity_model')
tokenizer.save_pretrained('./roberta_bug_severity_model')

# cell 8

from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Load the model and tokenizer
model = RobertaForSequenceClassification.from_pretrained('./roberta_bug_severity_model')
tokenizer = RobertaTokenizer.from_pretrained('./roberta_bug_severity_model')
model.to(device)

# Example inference on new data
def predict_bug_severity(code, code_comment):
    model.eval()
    combined_input = code + ' ' + code_comment
    inputs = tokenizer(combined_input, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    return prediction

# Example usage
code_example_0 = "public void printMessage() { System.out.println(\"Hello World\"); }"
comment_example_0 = "Simple print method. No known issues."
code_example_1 = """
public int divide(int a, int b) {
    if (b == 0) return -1; // Should return an exception, not -1
    return a / b;
}
"""
comment_example_1 = "Handles division but returns -1 instead of throwing an exception for divide by zero."
code_example_2 = """
public void processFile(String fileName) {
    File file = new File(fileName);
    file.delete(); // Deletes file without confirmation, risky operation
}
"""
comment_example_2 = "Deletes file without any user confirmation. Could lead to data loss."
code_example_3 = """
public void authenticateUser(String password) {
    if (password.equals(\"hardcoded_password\")) {
        System.out.println(\"Authentication successful\");
    }
}
"""
comment_example_3 = "Uses a hardcoded password, a serious security vulnerability."

predicted_severity_0 = predict_bug_severity(code_example_0, comment_example_0)
predicted_severity_1 = predict_bug_severity(code_example_1, comment_example_1)
predicted_severity_2 = predict_bug_severity(code_example_2, comment_example_2)
predicted_severity_3 = predict_bug_severity(code_example_3, comment_example_3)

print(f"Predicted Bug Severity: {predicted_severity_0}")
print(f"Predicted Bug Severity: {predicted_severity_1}")
print(f"Predicted Bug Severity: {predicted_severity_2}")
print(f"Predicted Bug Severity: {predicted_severity_3}")

# cell 9

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load scaled data
train_data = pd.read_csv('train_scaled.csv')
valid_data = pd.read_csv('valid_scaled.csv')
test_data = pd.read_csv('test_scaled.csv')

# Extract features and labels
X_train = train_data.drop(columns=['label', 'code', 'code_comment'])
y_train = train_data['label']
X_valid = valid_data.drop(columns=['label', 'code', 'code_comment'])
y_valid = valid_data['label']
X_test = test_data.drop(columns=['label', 'code', 'code_comment'])
y_test = test_data['label']

unique_classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

print("Class weight dictionary:", class_weight_dict)

# Train RandomForest
rf_model = RandomForestClassifier(class_weight=class_weight_dict)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Train DecisionTree
dt_model = DecisionTreeClassifier(class_weight='balanced')
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# Train SVM
svm_model = SVC(class_weight='balanced')
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# Calculate metrics for each model
models = ['RandomForest', 'DecisionTree', 'SVM']
predictions = [rf_preds, dt_preds, svm_preds]

metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

for preds in predictions:
    metrics['Accuracy'].append(accuracy_score(y_test, preds))
    metrics['Precision'].append(precision_score(y_test, preds, average='weighted'))
    metrics['Recall'].append(recall_score(y_test, preds, average='weighted'))
    metrics['F1 Score'].append(f1_score(y_test, preds, average='weighted'))

# Plot the metrics
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Comparison of Model Performance Metrics')

metric_names = list(metrics.keys())
for i, metric in enumerate(metric_names):
    ax[i // 2, i % 2].bar(models, metrics[metric], color=['blue', 'green', 'red'])
    ax[i // 2, i % 2].set_title(metric)
    ax[i // 2, i % 2].set_ylim(0, 1)
    ax[i // 2, i % 2].set_ylabel(metric)

plt.tight_layout()
plt.show()

# Print detailed classification reports for each model
print("RandomForest Classification Report:")
print(classification_report(y_test, rf_preds))

print("DecisionTree Classification Report:")
print(classification_report(y_test, dt_preds))

print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))
