import streamlit as st
import pandas as pd

st.title("E&C Chatbot Analytics Dashboard")

# Load processed data
df = pd.read_excel("../data/chat_data_with_topics.xlsx")

# Show summary
st.subheader("Messages per Topic")
topic_counts = df['Topic'].value_counts().sort_index()
st.bar_chart(topic_counts)

st.subheader("Sample Messages")
st.dataframe(df[['User Message', 'Bot Response', 'Topic']].head(20))

# Optional: add filters
topic_filter = st.selectbox("Select Topic", options=df['Topic'].unique())
st.subheader(f"Messages for Topic {topic_filter}")
st.dataframe(df[df['Topic'] == topic_filter][['User Message', 'Bot Response']])
