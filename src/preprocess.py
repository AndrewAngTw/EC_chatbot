import os
import fitz  # PyMuPDF
import docx
import pandas as pd

DOCS_PATH = "./data/docs/"

LABEL_KEYWORDS = {
    "whistle": "Whistleblowing",
    "conduct": "Code of Conduct",
    "training": "Training",
    "privacy": "Data Privacy",
    "anti-bribery": "Anti-Bribery",
    "conflict": "Conflicts of Interest",
    "security": "Information Security"
}

def read_pdf(path):
    # Extract text from PDF 
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text("text") + " "
    return text.strip()

def read_docx(path):
    # Extract text from DOCX files
    doc = docx.Document(path)
    return " ".join([para.text for para in doc.paragraphs])

def read_txt(path):
    # Read plain TXT files
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def assign_label(filename):
    # Assign label based on keywords in the filename
    filename_lower = filename.lower()
    for keyword, label in LABEL_KEYWORDS.items():
        if keyword in filename_lower:
            return label
    return "Other"

def load_documents():
    records = []
    for filename in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, filename)
        if filename.endswith(".pdf"):
            text = read_pdf(path)
        elif filename.endswith(".docx"):
            text = read_docx(path)
        elif filename.endswith(".txt"):
            text = read_txt(path)
        else:
            continue

        label = assign_label(filename)
        records.append({"filename": filename, "text": text, "label": label})

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = load_documents()
    df.to_excel("./data/labeled_docs.xlsx", index=False)
    print("Saved labeled documents to data/labeled_docs.xlsx")
