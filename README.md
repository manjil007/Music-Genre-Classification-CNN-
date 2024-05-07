# Logistic Regression Music Genre Classification

This repository contains a machine learning project aimed at classifying music genres using a custom implementation of the Logistic Regression algorithm. The project is structured into two main components: feature extraction and model training & evaluation.

## Getting Started

### Prerequisites
Ensure you have Python installed along with the following libraries:
audioread==3.0.1
certifi==2024.2.2
cffi==1.16.0
charset-normalizer==3.3.2
decorator==5.1.1
idna==3.7
joblib==1.4.0
lazy_loader==0.4
librosa==0.10.1
llvmlite==0.42.0
msgpack==1.0.8
numba==0.59.1
numpy==1.26.4
packaging==24.0
pandas==2.2.2
platformdirs==4.2.0
pooch==1.8.1
pycparser==2.22
python-dateutil==2.9.0.post0
pytz==2024.1
requests==2.31.0
scikit-learn==1.4.2
scipy==1.13.0
six==1.16.0
soundfile==0.12.1
soxr==0.3.7
threadpoolctl==3.4.0
typing_extensions==4.11.0
tzdata==2024.1
urllib3==2.2.1

torch~=2.3.0
matplotlib~=3.8.4
tensorflow~=2.16.1
seaborn~=0.13.2
pydub~=0.25.1

### Feature Extraction
Before training the model, features must be extracted from audio files. This is done using the `process_data.py` script and `audio_processor.py` script. 


### Model Training and Evaluation
Once the features are extracted and saved as CSV files and PNG files, the model can be trained using the train.py script. Before running the script, make sure the paths to the pre-extracted feature CSV files are correctly set. 

Modify these lines in train.py to use your feature CSV files
train_df = pd.read_csv('path/to/your/train_features.csv')
test_df = pd.read_csv('path/to/your/test_features.csv')

The script will train the Convolutional Neural Network model, evaluate its performance, and save the predictions to a CSV file.

### CNN Class
The CNN class captures the convolutional neural network algorithm tailored specifically to the task of genre classifictaion based on images generated. This CNN is designed to handle pre-processed spectrogram images for classification into multiple categories.

### MLP Class
The MLP class encapsulates the Multi-Layer Perceptron model using PyTorch framework. This type of neural network is particularly suited for classification tasks involving structured data.

### transfer_learning Class
The transfer_learning class is the application of transfer learning using the ResNet50 architecture, a pre-trained convolutional neural network provided by TensorFlow. This model is extensively adapted to perform image classification tasks on a new dataset that might have different classes from the dataset originally used to train ResNet50.

### Running the Scripts and Description

1. **Feature Extraction**: The python script 'process_data.py' contains all functions necessary for the feature extraction from audio files and to creae a CSV file that will be used by our model. audio_processor.py contains functions necessary for extracting images like spectrograms from the audio files.

2. **Training Model**: The python script 'train.py' contains all functions necessary for the training of our model and generate predictions. Also prints out the cross validation accuracy as well as create a prediction file which can be submitted in the Kaggle to check accuracy. 

3. **Convolutional Neural network**: The python script 'cnn.py' contains the algorithm for convolutional neural network. 

3. **Multi-Layer Perceptron**: The python script 'mlp.py' contains the algorithm for multi-layer perceptron.

3. **Transfer Learning**: The python script 'transfer_learning.py' contains the algorithm for transfer learning.


### Highest Kaggle Scores
Accuracy: 83% Leaderboard Position: 2 Date Submitted : 05/07/2024
Accuracy: 74% Leaderboard Position: 9 Date Submitted : 05/06/2024

### Contributions
Manjil - Worked on CNN, transfer learning, train, test, and evaluation. Worked on data visualization for the evaluation of the performance of the models. Also conducted experiments for model comparison and worked on data preprocessing, implementation of CNN, model evaluation, model comparision of the report. 


Vincent - Worked on mlp.py and optimizing MLP parameters for highest possible accuracy using provided MLP sample code as reference. Worked on both MLP and optimization sections in the report.

Erick - Worked on extracting spectrograms from audio files using Professor Estrada's sample code as a starting point. Also worked on extracting Chroma features and zero-crossing rate, though since they did not provide as high an accuracy they were discarded. Helped with MLP and Optimizations sections of the report.

Abhinav - Worked on transfer learning, audio processing, feature extraction, and evaluation. Also conducted experiments for model comparison and worked on transfer learning, model evaluation, model comparision, intro, and conclusion of the report. 

## Acknowledgments
Thanks to Professor. Estrada, and Teaching Assistants for their support during the project. Thanks to the our classmates and the machine learning community at large for the inspiration and support to develop this project.