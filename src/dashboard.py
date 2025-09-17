import streamlit as st
import pandas as pd
import joblib

st.title("E&C Chatbot Analytics Dashboard")

# Load classifier + vectorizer
clf = joblib.load("./data/classifier.pkl")
vectorizer = joblib.load("./data/vectorizer.pkl")

def classify_text(text):
    # Classify a single message into a topic
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]

# Upload or load chat data
uploaded_file = st.file_uploader("Upload chat data (Excel)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
else:
    st.info("Using default chat data")
    df = pd.read_excel("./data/chat_data_with_topics.xlsx")

# Classify messages if 'Topic' column doesn't exist
if 'Topic' not in df.columns:
    st.warning("Classifying messages dynamically...")
    df['Topic'] = df['User Message'].apply(classify_text)

# Show summary
st.subheader("Messages per Topic")
topic_counts = df['Topic'].value_counts().sort_index()
st.bar_chart(topic_counts)

st.subheader("Sample Messages")
st.dataframe(df[['User Message', 'Bot Response', 'Topic']].head(20))

# Filter by topic
topic_filter = st.selectbox("Select Topic", options=df['Topic'].unique())
st.subheader(f"Messages for Topic: {topic_filter}")
st.dataframe(df[df['Topic'] == topic_filter][['User Message', 'Bot Response']])
