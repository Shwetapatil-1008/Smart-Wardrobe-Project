import os  # ✅ Add this line
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load your trained model
model = load_model("FinalFab_fix_classifier.keras")

# Define fabric class labels
class_labels = ['Cotton', 'Denim', 'Silk', 'Wool']

# Path to test dataset
test_data_dir = "split_dataset/test"

# Lists for storing results
actual_labels = []
predicted_labels = []
y_true = []
y_pred_prob = []

# Loop through test dataset folders
for fabric_type in class_labels:  # Ensure class labels match dataset
    fabric_folder = f"{test_data_dir}/{fabric_type}"
    
    for img_file in os.listdir(fabric_folder):  
        img_path = f"{fabric_folder}/{img_file}"
        
        # Load & preprocess image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict probabilities & class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        
        # Store actual & predicted labels
        actual_labels.append(fabric_type)
        predicted_labels.append(class_labels[predicted_class])
        y_true.append(class_labels.index(fabric_type))
        y_pred_prob.append(predictions[0])

# Compute accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Model Accuracy: {accuracy:.2%}")

# Generate confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=class_labels)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix\nModel Accuracy: {accuracy:.2%}")
plt.show()

# Print Classification Report
print("Classification Report:\n", classification_report(actual_labels, predicted_labels, target_names=class_labels))


import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(x=predicted_labels, order=class_labels, palette='coolwarm')
plt.xlabel('Predicted Fabric Type')
plt.ylabel('Count')
plt.title('Fabric Type Prediction Distribution')
plt.show()



plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis], 
            annot=True, cmap='Blues', fmt=".2f", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix (Misclassification %)")
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Convert true labels to one-hot encoding
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])  

# ✅ Convert predictions to NumPy array
y_pred_prob = np.array(y_pred_prob)

# Plot Precision-Recall curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(class_labels)):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
    avg_precision = average_precision_score(y_true_bin[:, i], y_pred_prob[:, i])
    plt.plot(recall, precision, label=f"{class_labels[i]} (AP = {avg_precision:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multiclass Precision-Recall Curve")
plt.legend()
plt.show()




from sklearn.metrics import roc_curve, auc

# ✅ Convert predictions to NumPy array
y_pred_prob = np.array(y_pred_prob)

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(class_labels)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_labels[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend()
plt.show()


