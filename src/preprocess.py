import os
import fitz  # PyMuPDF
import docx
import pandas as pd

DOCS_PATH = "./data/docs/"

def read_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text("text") + " "
    return text.strip()

def read_docx(path):
    doc = docx.Document(path)
    return " ".join([para.text for para in doc.paragraphs])

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_documents():
    records = []

    # Loop over folders in DOCS_PATH
    for folder_name in os.listdir(DOCS_PATH):
        folder_path = os.path.join(DOCS_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue  # skip files in the root folder

        # folder_name is the label
        label = folder_name

        # Loop over files in folder
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)

            # Read content
            if filename.endswith(".pdf"):
                text = read_pdf(path)
            elif filename.endswith(".docx"):
                text = read_docx(path)
            elif filename.endswith(".txt"):
                text = read_txt(path)
            else:
                continue  # skip unsupported files

            records.append({
                "filename": filename,
                "text": text,
                "label": label
            })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = load_documents()
    df.to_excel("./data/labeled_docs.xlsx", index=False)
    print(f"✅ Saved {len(df)} labeled documents to data/labeled_docs.xlsx")

