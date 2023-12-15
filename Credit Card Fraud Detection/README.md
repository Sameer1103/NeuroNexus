# Credit Card Transaction Fraud Detection

### Overview
This project focuses on classifying the credit card transactions as fraudulent or legitimate based on the transaction details using different machine learning classification models, including Logistic Regression, Decision Tree and Random Forest. Each model is trained on the same dataset and their accuracy is compared.

### Dataset
The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) obtained from Kaggle. This contains both training and testing dataset consisting of merchant details and credit-card holder details.

### Data Preprocessing
Comprehensive data preprocessing steps are used such as calculating age from transaction date, calculating distance between credit-card holder and merchant. The redundant columns are removed and the gender and category values are encoded to obtain numeric data. The transactions are split into training and testing datasets in the ratio of 3:1.

### Machine Learning Models
Three classification models are implemented and compared:

1. Logistic Regression
1. Decision Tree
1. Random Forest

Models are trained on the training set, and their accuracies are evaluated on both training and testing datasets.

### Model Testing
Trained models are first tested on sample transaction details to demonstrate their predictive capabilities. Then they are tested on test dataset to measure accuracy. 

### Results
Accuracy scores and Classification Report are displayed in the notebook. The Random Forest Classifier achieved the highest accuracy among all the tested models.

### How to Use
1. Open CC Fraud Detection.ipynb file in Jupyter Notebook.
1. Download the dataset from kaggle and move it to same folder as notebook file.
1. Execute cells to load data, preprocess it, train models, and make predictions.
