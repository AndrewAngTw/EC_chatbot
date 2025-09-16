import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from preprocess import load_data, preprocess

# Load and preprocess
df = load_data("../data/chat_data.xlsx")
df = preprocess(df)

# Vectorize
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['cleaned_message'])

# Fit LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Assign topics to each message
topic_values = lda.transform(X)
df['Topic'] = topic_values.argmax(axis=1)

# Show top words per topic
words = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx+1}: {[words[i] for i in topic.argsort()[-10:]]}")

# Save results
df.to_excel("../data/chat_data_with_topics.xlsx", index=False)

