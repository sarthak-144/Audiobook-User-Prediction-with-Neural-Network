# Audiobook User Prediction with Neural Networks

## Overview

This project builds a machine learning model using TensorFlow to predict audiobook user behavior based on historical data. The dataset includes various numerical features representing user interactions with audiobooks.

## Dataset

- **Source:** Audiobooks\_data.csv
- **Size:** 14,083 entries
- **Features:** 12 numerical columns (user behavior-related data)

## Model Architecture

- **Framework:** TensorFlow/Keras
- **Layers:**
  - Three hidden layers with ReLU activation
  - Output layer with softmax activation (2 classes)
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam

## Usage

1. Prepare the dataset and load it into numpy arrays:
   ```python
   npz = np.load('data_train.npz')
   train_inputs = npz['inputs'].astype(np.float32)
   train_targets = npz['targets'].astype(np.int32)
   ```
2. Train the model:
   ```python
   model.fit(train_inputs, train_targets, batch_size=32, epochs=100, validation_data=(validation_inputs, validation_targets))
   ```
3. Evaluate on the test set:
   ```python
   test_loss, test_acc = model.evaluate(test_inputs, test_targets)
   print(f'Test Accuracy: {test_acc}')
   ```

## Results

The trained model predicts user behavior with a certain accuracy based on audiobook usage patterns. Performance metrics are logged during training.

- **Accuracy:** 0.8379
- **Loss:** 0.3078
