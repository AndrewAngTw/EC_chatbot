import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_classifier():
    # Load labeled training documents
    df = pd.read_excel("./data/labeled_docs.xlsx")

    # Clean data
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]

    X = df["text"]
    y = df["label"]

    min_class_size = y.value_counts().min()

    if len(y.unique()) > 1 and min_class_size > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Convert text → TF-IDF features
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    # Evaluate if multiple labels exist
    if len(y.unique()) > 1:
        preds = clf.predict(X_test_vec)
        print("Classification Report:")
        print(classification_report(y_test, preds))
    else:
        print("Only one class found in data — skipping evaluation")

    # Save model + vectorizer
    joblib.dump(clf, "./data/classifier.pkl")
    joblib.dump(vectorizer, "./data/vectorizer.pkl")
    print("Saved classifier.pkl and vectorizer.pkl")



def extract_doc_name(bot_response: str) -> str:
    # Match the last quoted text at the end
    matches = re.findall(r'"([^"]*)"$', bot_response.strip())
    return matches[0] if matches else None

def classify_chat_data():
    chat_df = pd.read_excel("./data/chat_data.xlsx")

    # Load classifier + vectorizer (optional if you still want text-based classification)
    clf = joblib.load("./data/classifier.pkl")
    vectorizer = joblib.load("./data/vectorizer.pkl")

    # Ensure no NaN values
    chat_df['User Message'] = chat_df['User Message'].fillna("")
    chat_df['Bot Response'] = chat_df['Bot Response'].fillna("")

    # Extract filename
    chat_df['filename'] = chat_df['Bot Response'].apply(extract_doc_name)

    # Load labeled docs
    labeled_df = pd.read_excel("./data/labeled_docs.xlsx")

    # Create a dictionary: filename → label
    doc_to_label = dict(zip(labeled_df['filename'], labeled_df['label']))

    # Map filename → label
    chat_df['Topic'] = chat_df['filename'].map(doc_to_label)

    # Optionally fallback to text classifier if filename not found
    chat_df['Topic'] = chat_df.apply(
        lambda row: row['Topic'] if pd.notna(row['Topic']) 
        else (clf.predict(vectorizer.transform([row['User Message']]))[0] if row['User Message'].strip() else "Other"),
        axis=1
    )

    chat_df.to_excel("./data/chat_data_with_topics.xlsx", index=False)
    print("Updated chat_data_with_topics.xlsx")

if __name__ == "__main__":
    train_classifier()
    classify_chat_data()

