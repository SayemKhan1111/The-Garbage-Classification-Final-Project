import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas

# ⚙️ Config
image_size = (224, 224)
batch_size = 32
epochs = 20
data_dir = "data"

# 🔄 Preprocessing with EfficientNet
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# 🔧 Load EfficientNetV2B2 base
base_model = EfficientNetV2B2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False  # Freeze base initially

# 🧠 Final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')
])

# 🛠 Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 📉 Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# ▶️ Train the model
history = model.fit(
    train,
    validation_data=val,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr]
)

# 💾 Save model
os.makedirs("models", exist_ok=True)
model.save("models/efficientnetv2b2_model.keras")

# 🔄 Fine-tune the model
base_model.trainable = True  # Unfreeze the base model for fine-tuning

# It's important to freeze the earlier layers before fine-tuning
for layer in base_model.layers[:100]:  # Fine-tune later layers only
    layer.trainable = False

# Re-compile the model after unfreezing
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ▶️ Fine-tune the model
history_finetune = model.fit(
    train,
    validation_data=val,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr]
)

# 📊 Evaluation: Confusion Matrix & Classification Report
test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=1,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# -------- 1. Confusion Matrix (Heatmap) --------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------- 2. Classification Report (Bar Chart) --------
report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys(), output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot Precision, Recall, F1-Score for each class
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6))
plt.title('Classification Report: Precision, Recall, F1-Score')
plt.ylabel('Score')
plt.xlabel('Classes')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# -------- 3. Training and Validation Accuracy (Line Chart) --------
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# -------- 4. Training and Validation Loss (Line Chart) --------
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# -------- 5. Histogram of Class Distributions (Bar Chart) --------
class_counts = np.bincount(y_true)  # Count number of samples per class

plt.bar(test_generator.class_indices.keys(), class_counts)
plt.title('Class Distribution in Dataset')
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------- 6. Histogram of Predictions (Class Distribution in Predictions) --------
predicted_class_counts = np.bincount(y_pred)

plt.bar(test_generator.class_indices.keys(), predicted_class_counts)
plt.title('Predicted Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Number of Predictions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------- 7. Loss vs Accuracy (Combined Chart) --------
fig, ax1 = plt.subplots()

# Plot accuracy on the first axis
ax1.plot(history.history['accuracy'], color='tab:blue', label='Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second axis to plot loss
ax2 = ax1.twinx()
ax2.plot(history.history['loss'], color='tab:red', label='Loss')
ax2.set_ylabel('Loss', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Accuracy vs Loss')
plt.show()

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
