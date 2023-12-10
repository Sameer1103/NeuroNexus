# MOVIE GENRE CLASSIFICATION
### Overview
This project focuses on classifying the movies based on their genres using different machine learning classification models, including Logistic Regression, Naive Bayes, and Support Vector Machines (SVM). Each model is trained on the same dataset and their accuracy is compared.

### Dataset
The dataset used is the [Genre Classification Dataset IMDb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb) obtained from Kaggle. This contains both training and testing dataset consisting of movie title and plot summary.

### Data Preprocessing
Comprehensive data preprocessing steps are used, such as converting to lowercase, removing special characters and punctuation and removing extra whitespaces. The redundant columns are removed and the movie genres are encoded to obtain numeric data. The movies are split into training and testing datasets in the ratio of 3:1.

### Feature Extraction
Text data is transformed into feature vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer, enabling numerical input for machine learning models.

### Machine Learning Models
Three classification models are implemented and compared:

1. Multinomial Naive Bayes
1. Logistic Regression
1. Support Vector Classification (SVC)

Models are trained on the training set, and their accuracies are evaluated on both training and testing datasets.

### Model Testing
Trained models are tested on sample movie plot summaries to demonstrate their predictive capabilities. Input plot summaries are converted into feature vectors using the TF-IDF vectorizer.

### Results
Accuracy scores and Classification Report are displayed in the notebook. The Logistic Regression Classifier achieved the highest accuracy among all the tested models.

### How to Use
1. Open Genre_Classification.ipynb file in Jupyter Notebook.
1. Download the dataset from kaggle and move it to same folder as notebook file.
1. Execute cells to load data, preprocess it, train models, and make predictions.

