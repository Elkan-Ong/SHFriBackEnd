import pickle
import re
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#Download NLTK stopwords if not already done
nltk.download('stopwords')

def clean_text(text):
    """
    Cleans the input text by:
    - Converting to lowercase
    - Removing special characters and digits
    - Removing stopwords
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize the text into words
    words = text.split()

    # Remove stopwords (from NLTK stopwords + extra stopwords)
    stop_words = set(stopwords.words('english'))  # NLTK stopwords

    # Filter out stopwords
    cleaned_text = ' '.join([word for word in words if word not in stop_words])

    return cleaned_text


class FakeNewsModel:
    def __init__(self, model=None, clean_func=None):
        self.model = model
        self.clean_func = clean_func

    def train(self, df):
        """
        Trains the model using the given dataframe.
        """
        # Clean the text in the dataframe
        df['clean_text'] = df['text'].apply(self.clean_func)

        # Train the model
        X = df['clean_text']
        y = df['target']

        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression())
        ])

        self.model.fit(X, y)

    def predict_fake_news(self, text):
        """
        Returns the probability that the input text is fake news.
        """
        # Clean the input text
        cleaned_text = self.clean_func(text)

        # Get the probability of the text being fake (class 1)
        prob_fake = self.model.predict_proba([cleaned_text])[0, 1]

        return prob_fake

    def save_model(self, filename='fake_news_model.pkl'):
        """
        Save the model and pre-processing function to a pickle file.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model and pre-processing function saved to {filename}")

    @staticmethod
    def load_model(filename='fake_news_model.pkl'):
        """
        Load the model and pre-processing function from a pickle file.
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model and pre-processing function loaded from {filename}")
        return model