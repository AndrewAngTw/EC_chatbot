import pandas as pd
import re

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(df):
    df['cleaned_message'] = df['User Message'].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_data("../data/chat_data.xlsx")
    df = preprocess(df)
    print(df.head())
