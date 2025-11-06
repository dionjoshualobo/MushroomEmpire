import csv
import re
from pathlib import Path

ROOT = Path("../../archiv/Data/Politics")

email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
phone_re = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")
ssn_re = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
uuid_re = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
pan_re = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
iban_re = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b")

patterns = {
    "EMAIL": email_re,
    "PHONE": phone_re,
    "SSN": ssn_re,
    "UUID": uuid_re,
    "PAN": pan_re,
    "IBAN": iban_re,
}

def find_entities(text):
    found = []
    for label, pattern in patterns.items():
        for m in pattern.finditer(text):
            found.append(f"{label}: {m.group(0)}")
    return found

with open("discovery_dataset.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "text", "detected_entities"])

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        entities = find_entities(text)
        if text.strip():  # skip empty
            writer.writerow([str(path), text[:5000], "; ".join(entities)])  # limit length if huge
