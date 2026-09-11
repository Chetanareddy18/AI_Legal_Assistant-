import json
import os

import pandas as pd
from dotenv import load_dotenv
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import (
    EntitiesOptions,
    Features,
    KeywordsOptions,
)
from PyPDF2 import PdfReader

load_dotenv()
API_KEY = os.getenv("IBM_NLU_API_KEY")
URL = os.getenv("IBM_NLU_URL")

authenticator = IAMAuthenticator(API_KEY)
nlu = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator
)
nlu.set_service_url(URL)

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_text(text):
    response = nlu.analyze(
        text=text,
        features=Features(
            entities=EntitiesOptions(limit=10),
            keywords=KeywordsOptions(limit=10)
        )
    ).get_result()
    return response

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(base, "data")
    results_folder = os.path.join(base, "results")
    parquet_path = os.path.join(results_folder, "preprocess_output.parquet")
    csv_path = os.path.join(results_folder, "preprocess_output.csv")

    os.makedirs(results_folder, exist_ok=True)

    rows = []

    for file_name in os.listdir(data_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(data_folder, file_name)

            text = extract_text_from_pdf(file_path)
            nlu_output = analyze_text(text)

            text_path = os.path.join(results_folder, file_name.replace(".pdf", "_text.txt"))
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            nlu_path = os.path.join(results_folder, file_name.replace(".pdf", "_nlu.json"))
            with open(nlu_path, "w", encoding="utf-8") as f:
                json.dump(nlu_output, f, indent=4)

            full_obj = {
                "file": file_name,
                "text": text,
                "nlu": nlu_output
            }
            full_path = os.path.join(results_folder, file_name.replace(".pdf", "_full.json"))
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(full_obj, f, indent=4)
            rows.append({
                "file": file_name,
                "text": text,
                "nlu": json.dumps(nlu_output)
            })
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")
