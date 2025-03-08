import os
import shutil
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pytesseract
from pdf2image import convert_from_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF, using different PyMuPDF extraction methods."""
    try:
        doc = fitz.open(pdf_path)

        # Use 'text' mode first
        text = "\n".join([page.get_text("text") for page in doc])

        # Try 'blocks' if 'text' mode fails
        if not text.strip():
            text = "\n".join([str(page.get_text("blocks")) for page in doc])

        # Try 'words' if still empty
        if not text.strip():
            text = " ".join([w[4] for page in doc for w in page.get_text("words")])

        # OCR as a last resort
        if not text.strip():
            print(f"OCR applied to: {pdf_path}")
            images = convert_from_path(pdf_path)
            text = "\n".join(pytesseract.image_to_string(img) for img in images)

        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def load_dataset(dataset_dir):
    """Loads classified PDFs from dataset directory."""
    data, labels = [], []
    categories = os.listdir(dataset_dir)

    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(category_path, filename)
                    text = extract_text_from_pdf(pdf_path)
                    if text.strip():
                        data.append(text)
                        labels.append(category)
    return data, labels


def train_model(data, labels):
    model = make_pipeline(TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2)), RandomForestClassifier(n_estimators=300, max_depth=50))
  #  model = make_pipeline(TfidfVectorizer(stop_words='english', max_features=15000, ngram_range=(1,2)), LogisticRegression())
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            print(f"Misclassified: Expected {y_test[i]}, but got {y_pred[i]}")

    return model


def classify_and_move_pdfs(model, input_dir, output_dir):
    """Classifies PDFs and moves them to a separate classified output folder."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                category = model.predict([text])[0]
                category_dir = os.path.join(output_dir, category)  # Use output_dir instead of dataset_dir
                os.makedirs(category_dir, exist_ok=True)  # Create category folder if needed
                shutil.move(pdf_path, os.path.join(category_dir, filename))  # Move file to output directory
                print(f"Moved {filename} to {category_dir}")
            else:
                print(f"Skipping {filename}, no readable text.")



def main():
    dataset_dir = "dataset"
    input_dir = "Input"
    output_dir = "Output"  # New directory for classified PDFs

    print("Loading dataset...")
    data, labels = load_dataset(dataset_dir)

    print("Training model...")
    model = train_model(data, labels)

    print("Classifying new PDFs...")
    classify_and_move_pdfs(model, input_dir, output_dir)  # Send classified PDFs to output_dir



if __name__ == "__main__":
    main()
