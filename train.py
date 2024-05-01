# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import Input
# from sklearn.metrics import classification_report, confusion_matrix
#
#
# train_df = pd.read_csv('train_features.csv')
# test_df = pd.read_csv('test_features.csv')
#
# # Training data
# X_train = train_df.drop(['Genre'], axis=1).values
# y_train = train_df['Genre'].values
#
# # Test data
# X_test = test_df.drop(['id'], axis=1).values
# test_ids = test_df['id'].values
#
# # Encode labels
# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)
#
# # OneHot encode the labels
# onehot_encoder = OneHotEncoder()
# y_train_onehot = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1))
# # Convert sparse matrix to dense
# y_train_onehot_dense = y_train_onehot.toarray()
#
# # Feature scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # PCA for dimensionality reduction
# pca = PCA(n_components=0.99)
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)
# print("number of features after pca = ", X_train_pca.shape[1])
#
#
# # Define the KFold cross-validation setup
# kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Using 5 folds
# cv_scores = []
#
# # Iterate over each split
# for train_index, val_index in kf.split(X_train_pca):
#     X_cv_train, X_cv_val = X_train_pca[train_index], X_train_pca[val_index]
#     y_cv_train, y_cv_val = y_train_onehot_dense[train_index], y_train_onehot_dense[val_index]
#
#     # Define the model architecture
#     model = Sequential([
#         Input(shape=(X_train_pca.shape[1],)),
#         Dense(256, activation='relu'),
#         Dropout(0.3),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         Dense(y_cv_train.shape[1], activation='softmax')
#     ])
#
#     # Compile the model
#     model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#
#     # Train the model
#     history = model.fit(
#         X_cv_train,
#         y_cv_train,
#         epochs=50,
#         batch_size=32,
#         validation_data=(X_cv_val, y_cv_val),
#         callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
#     )
#
#     # Evaluate the model on the validation fold
#     val_loss, val_accuracy = model.evaluate(X_cv_val, y_cv_val)
#     cv_scores.append(val_accuracy)
#     print(f'Fold validation accuracy: {val_accuracy}')
#
# # Report average accuracy across all folds
# average_cv_accuracy = np.mean(cv_scores)
# print(f'Average CV Accuracy: {average_cv_accuracy}')
#
# # If the cross-validation performance is satisfactory, retrain on the entire dataset
# model.fit(X_train_pca, y_train_onehot_dense, epochs=50, batch_size=32,
#           callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
#
# # Predict on the test set for Kaggle submission
# test_predictions = model.predict(X_test_pca)
# test_predicted_labels = label_encoder.inverse_transform(np.argmax(test_predictions, axis=1))
#
# predictions_df = pd.DataFrame({'id': test_ids, 'predicted_genre': test_predicted_labels})
# predictions_output_path = 'test_predictions.csv'
# predictions_df.to_csv(predictions_output_path, index=False)
# print(f'Test predictions saved to {predictions_output_path}')
#
#


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
train_df = pd.read_csv('train_features.csv')
test_df = pd.read_csv('test_features.csv')

# Prepare training and testing data
X_train = train_df.drop(['Genre'], axis=1).values.reshape(-1, train_df.shape[1] - 1, 1)  # Reshape for CNN
y_train = train_df['Genre'].values
X_test = test_df.drop(['id'], axis=1).values.reshape(-1, test_df.shape[1] - 1, 1)  # Reshape for CNN
test_ids = test_df['id'].values

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# OneHot encode the labels
onehot_encoder = OneHotEncoder()
y_train_onehot = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1)).toarray()

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train_onehot, test_size=0.2, random_state=42)

# Model architecture
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {val_accuracy}')

# Predict on the test set for Kaggle submission
test_predictions = model.predict(X_test_scaled)
test_predicted_labels = label_encoder.inverse_transform(np.argmax(test_predictions, axis=1))

predictions_df = pd.DataFrame({'id': test_ids, 'predicted_genre': test_predicted_labels})
predictions_output_path = 'test_predictions.csv'
predictions_df.to_csv(predictions_output_path, index=False)
print(f'Test predictions saved to {predictions_output_path}')

