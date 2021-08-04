import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read dataset
csv_path = 'news.csv'
df=pd.read_csv(csv_path)

# Get labels
labels=df.label

# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform train set and transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
def test_detectotr():
    y_pred=pac.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    print(f'Accuracy: {round(score*100,2)}%')

    # Build confusion matrix
    confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

def test_news():
    print("Put your sentence below:")
    input_data = [input()]
    vectorized_input_data = tfidf_vectorizer.transform(input_data)
    prediction = pac.predict(vectorized_input_data)
    print(prediction)

#test_detectotr()
#test_news()