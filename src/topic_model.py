import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_classifier():
    # Load labeled documents
    df = pd.read_excel("./data/labeled_docs.xlsx")

    # Remove empty or NaN text entries
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    # X = text, y = topic label (string)
    X = df["text"]
    y = df["label"] 

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert text → TF-IDF features
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)  

    # Evaluate
    preds = clf.predict(X_test_vec)
    print("Classification Report:\n")
    print(classification_report(y_test, preds))

    # Save model + vectorizer
    joblib.dump(clf, "./data/classifier.pkl")
    joblib.dump(vectorizer, "./data/vectorizer.pkl")
    print("Classifier and vectorizer saved to ./data/")

if __name__ == "__main__":
    train_classifier()
