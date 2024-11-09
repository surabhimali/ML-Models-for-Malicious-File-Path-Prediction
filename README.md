# ML Models for Malicious File Path Prediction
 Developing a machine learning model to accurately predict and classify file paths as malicious or benign, enhancing cybersecurity measures and threat detection.

## Data Pre-processing

### Overview
This project involves data pre-processing tasks to prepare the data for machine learning model training. It includes loading, handling, cleaning, and transforming the data.


1. **Data Pre-processing:**
   - Libraries: The code starts by importing necessary libraries such as pandas and csv for data handling and manipulation.
   - File Paths: It defines file paths for both malicious and benign datasets, along with an output CSV file path for storing preprocessed data.
   - Reading and Handling Data Files: The code reads data from the malicious and benign files, handling file not found errors and loading the data into separate lists.
   - Data Extraction and Transformation: Information is extracted from file paths, processed, and stored in a list of lists for further processing.
   - DataFrame Creation and Inspection: The extracted data is used to create a pandas DataFrame with appropriate column names. The code checks for missing values and duplicate rows in the DataFrame.

2. **Data Cleaning and Transformation:**
   - Duplicate rows in the DataFrame are handled by removing them if necessary.
   - The cleaned and transformed data is saved to a CSV file for further analysis and modeling.

---

## Encoding

### Overview
This section focuses on encoding categorical variables to prepare them for machine learning model training.

1. **Loading Data and Importing Libraries:**
   - The code imports necessary libraries such as pandas and LabelEncoder from sklearn.preprocessing.
2. **Loading the CSV File:**
   - It loads the preprocessed CSV file containing encoded data into a pandas DataFrame.
3. **Encoding Categorical Variables:**
   - LabelEncoder is used to encode categorical columns such as 'Classification', 'File Path', 'File Name', and 'File Type'.
4. **Saving the Encoded DataFrame:**
   - The encoded DataFrame is saved to a new CSV file for further use in model training and evaluation.

---
## Data Splitting for Machine Learning Model Training and Testing
1. **Train-Test Split for Benign and Malicious Data:**
   - The benign data (benign_df) is split into train (80%) and test (20%) sets using train_test_split with a random state for reproducibility.
   - Similarly, the malicious data (malicious_df) is split into train and test sets with the same proportions and random state.

2. **Separating Features and Target Variables:**
   - Features (X) and the target variable (y) are separated for both the training (train_data) and testing (test_data) sets.
   - Features are obtained by dropping the 'Classification' column, and the target variable is extracted as the 'Classification' column.

3. **Saving Train and Test Data to CSV Files:**
   - The train and test data are saved to separate CSV files (X_train.csv, X_test.csv, y_train.csv, y_test.csv) using to_csv() with the index parameter set to False to exclude the index column from the saved files.

---
## Machine Learning Models

### Random Forest Classifier

#### Overview
This part covers training, evaluating, and caching a Random Forest Classifier for predicting classifications.


- **Model Training and Evaluation with Random Forest Classifier:**
   - Model Loading: Checks for a cached model; if not found, proceeds with training a new model.
   - Data Loading: Loads training and testing data from CSV files.
   - Model Initialization: Initializes a Random Forest Classifier with specified parameters.
   - Model Training: Trains the classifier using the training data.
   - Model Caching: Saves the trained model for future use.
   - Model Prediction: Uses the trained model to make predictions on test data.
   - Performance Metrics Calculation: Calculates accuracy, confusion matrix, precision, recall, and F1 score.
   - Output Display: Prints the calculated metrics for the Random Forest Classifier.
   - Confusion Matrix Visualization: Plots and displays the confusion matrix.

### Gradient Boosting Classifier

#### Overview
This section covers training, evaluating, and caching a Gradient Boosting Classifier for classification tasks.

1. **Model Initialization and Training:**
   - Initializes a Gradient Boosting Classifier with specified parameters.
   - Trains the classifier using the training data.
2. **Model Caching:**
   - Saves the trained model for future use.
3. **Model Prediction and Performance Evaluation:**
   - Makes predictions on test data and calculates accuracy, confusion matrix, precision, recall, and F1 score.
4. **Confusion Matrix Visualization:**
   - Plots and displays the confusion matrix.

### Neural Network Classifier

#### Overview
This part focuses on training and evaluating a Neural Network Classifier using TensorFlow and Keras.

1. **Data Loading and Preprocessing:**
   - Loads the CSV files into DataFrames and encodes the target variable.
2. **Model Initialization and Training:**
   - Initializes a Sequential neural network model with input, hidden, and output layers.
   - Compiles and trains the model using training data.
3. **Model Evaluation:**
   - Evaluates the model using accuracy metrics and confusion matrix.
4. **Confusion Matrix Visualization:**
   - Plots and displays the confusion matrix.


## Summary and Conclusion

This README provides detailed insights into the data pre-processing, encoding, model training, evaluation in the machine learning project. Each section covers specific tasks, steps, and outcomes, showcasing the comprehensive approach to building and evaluating machine learning models for classification tasks.

