import logging
import pandas as pd
import re
from joblib import load
from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.client_context import ClientContext
import azure.functions as func
import json

# SharePoint settings
SITE_URL = "https://sats1.sharepoint.com/sites/Legal_chatbot"
PROCESSED_LIST_NAME = "ChatDataWithTopics"
CLIENT_ID = "<YOUR_CLIENT_ID>"
CLIENT_SECRET = "<YOUR_CLIENT_SECRET>"

# Load classifier and vectorizer
clf = load("classifier.pkl")
vectorizer = load("vectorizer.pkl")
labeled_df = pd.read_excel("labeled_docs.xlsx")
doc_to_label = dict(zip(labeled_df['filename'], labeled_df['label']))

def extract_doc_name(bot_response: str) -> str:
    matches = re.findall(r'"([^"]*)"$', bot_response.strip())
    return matches[0] if matches else None

def classify_message(user_message: str, bot_response: str) -> str:
    filename = extract_doc_name(bot_response)
    topic = doc_to_label.get(filename)
    if topic:
        return topic
    if user_message.strip():
        return clf.predict(vectorizer.transform([user_message]))[0]
    return "Other"

def push_to_sharepoint(user_message, bot_response, topic, filename):
    credentials = ClientCredential(CLIENT_ID, CLIENT_SECRET)
    ctx = ClientContext(SITE_URL).with_credentials(credentials)
    processed_list = ctx.web.lists.get_by_title(PROCESSED_LIST_NAME)
    item_creation_info = {
        "Title": user_message[:255],  # required column
        "UserMessage": user_message,
        "BotResponse": bot_response,
        "Topic": topic,
        "Filename": filename
    }
    processed_list.add_item(item_creation_info)
    ctx.execute_query()

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        user_message = req_body.get("user_message", "")
        bot_response = req_body.get("bot_response", "")
        topic = classify_message(user_message, bot_response)
        filename = extract_doc_name(bot_response)
        push_to_sharepoint(user_message, bot_response, topic, filename)
        return func.HttpResponse(json.dumps({"status":"success", "topic":topic}), status_code=200)
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(json.dumps({"status":"error","message":str(e)}), status_code=500)
