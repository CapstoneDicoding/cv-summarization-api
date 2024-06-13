import os
import re
import json
import pytesseract
import numpy as np
from google.cloud import storage
from fpdf import FPDF
from groq import Groq
import fitz
import logging
import urllib.parse
from flask import Flask, request
from PIL import Image
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

json_key_path = './key.json'

credentials = service_account.Credentials.from_service_account_file(
    json_key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

def download_pdf_from_storage(bucket_name, file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    pdf_bytes = blob.download_as_bytes()
    return pdf_bytes

def extract_data_from_pdf(pdf_bytes):
    extracted_text = ""
    doc = fitz.Document(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        custom_config = r'--psm 1'
        text = pytesseract.image_to_string(img, config=custom_config)
        extracted_text += text
    return extracted_text

def preprocessing(text):
    text = text.lower().strip()
    text = re.sub(r'\be\b', '', text)
    text = text.replace("'", "").replace('"', '')
    text = re.sub(r'[\n;,\|]', ' ', text)
    text = re.sub(r'\$1', 'S1', text)
    text = re.sub(r'\bpage\s\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'@gmail\s?com', '@gmail.com', text)
    text = re.sub(r'\s+@', '@', text)
    text = re.sub(r'@gmail\.co\b', '@gmail.com', text)
    text = re.sub(r'\+62\s?(\d{2,3})[ -]?(\d{3,4})[ -]?(\d{4,5})', r'+62\1\2\3', text)
    text = re.sub(r'08(\d{2})-(\d{4})-(\d{4})', r'+628\1\2\3', text)
    text = re.sub(r'\(\+62\)\s?(\d{2,3})[ -]?(\d{3,4})[ -]?(\d{4,5})', r'+62\1\2\3', text)
    text = re.sub(r'\(62\)\s?(\d{2,3})[ -]?(\d{3,4})[ -]?(\d{4,5})', r'+62\1\2\3', text)
    text = re.sub(r'\+62(\d+)', r'08\1', text)
    text = re.sub(r'https\s+(\w+)\s+com', r'https://\1.com', text)
    text = re.sub(r'(\d{4})\s*[~=_]\s*(present)', r'\1 - \2', text)
    text = re.sub(r'(\d{4})\s*[~=_]\s*(\w+)\s*(\d{4})?', r'\1 - \2 \3', text)
    text = re.sub(r'(\d{4})\s*[_=]\s*(\w+)', r'\1 - \2', text)
    words = text.split()
    return ' '.join(words)

def summarize_cv(cv_text):
    json_template = """
    {
        "basic_info": {  // Basic Information about the candidate
            "name": "<string>",  // Full name of the candidate
            "email": "<string>",  // Email address of the candidate
            "phone_number": "<string>",  // Phone number of the candidate
            "location": "<string>"  // Location of the candidate
        },
        "work_experience": [  // Array of work experience objects
            {
            "job_title": "<string>",  // Job title of the work experience
            "company": "<string>",  // Company of the work experience
            "location": "<string>",  // Location of the work experience
            "start_date": "<string>",  // Start date of the work experience (please convert the date into %b %Y format)
            "end_date": "<string>",  // End date of the work experience (please convert the date into %b %Y format)
            "job_desc": ["<string>", ...]  // Array of job descriptions ex. ["job_desc 1", "job_desc 2"]
            }
        ],
        "education": [  // Array of education objects
            {
            "title": "<string>",  // Title of the education (e.g. Bachelor's degree)
            "institute": "<string>",  // Institute of the education
            "location": "<string>",  // Location of the institute
            "start_date": "<string>",  // Start date of the education (please convert the date into %b %Y format)
            "end_date": "<string>",  // End date of the education (please convert the date into %b %Y format)
            "description": "<string>"  // Description of the education
            }
        ],
        "languages": ["<string>", ...],  // Array of languages spoken by the candidate
        "skills": ["<string>", ...],  // Array of skills possessed by the candidate
        "certification": [  // Array of certification objects
            {
            "title": "<string>",  // Title of the certification
            "issuer": "<string>",  // Issuer of the certification
            "start_date": "<string>",  // Start date of the certification (please convert the date into %b %Y format)
            "expiration_date": "<string>"  // Expiration date of the certification (please convert the date into %b %Y format)
            }
        ]
    }
    """

    client = Groq(api_key="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")

    prompt = f"""
    Extract the information from:
    \n{cv_text}\n. 
    
    Return the information in structured JSON format like the template below, strictly using only the data provided in the document. If any information is not mentioned in the CV, write 'Unknown'.

    \n{json.dumps(json_template, indent=4)}\n
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content":f"You are a JSON converter which receives raw CV candidate information as a string and returns a structured JSON output by organising the information in the string same as {json_template}."},
                   {"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0,
    )
    response = chat_completion.choices[0].message.content
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        json_str = response[json_start:json_end]
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        logging.error("Failed to parse JSON: %s", e)
        logging.error("Raw output: %s", response)
        return None

def upload_pdf_to_gcs(file_path, file_name):
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket("dicoding-capstone-bucket")
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_path)

def get_gcs_file_url(file_name):
    return f"https://storage.googleapis.com/dicoding-capstone-bucket/{file_name}"

def summarize_to_pdf(json_output, file_name):
    class PDF(FPDF):
        def header(self):
            self.set_font('Times', 'B', 14)
            self.cell(0, 7, 'CV Summary', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Times', '', 10)
            self.cell(0, 7, f'{self.page_no()}', 0, 0, 'C')

        def add_section_title(self, title):
            self.set_font('Times', 'BU', 12)
            self.cell(0, 5, title, 0, 1)
            self.ln(3)

        def add_section_subtitle(self, subtitle):
            self.set_font('Times', 'B', 12)
            self.cell(0, 5, subtitle, 0, 1)
            self.ln(2)

        def add_text(self, text):
            self.set_font('Times', '', 12)
            self.multi_cell(0, 5, text)
            self.ln(2)

    pdf = PDF()
    pdf.add_page()

    pdf.add_section_title("Personal Information")
    basic_info = json_output['basic_info']
    pdf.add_text(f"Name: {basic_info['name']}")
    pdf.add_text(f"Email: {basic_info['email']}")
    pdf.add_text(f"Phone Number: {basic_info['phone_number']}")
    pdf.add_text(f"Location: {basic_info['location']}")
    pdf.ln(3)

    pdf.add_section_title("Work Experience")
    for work in json_output['work_experience']:
        pdf.add_section_subtitle(f"{work['job_title']} | {work['company']}")
        pdf.add_text(f"Location: {work['location']}")
        pdf.add_text(f"Duration: {work['start_date']} - {work['end_date']}")
        pdf.add_text("Job Summary:")
        for job_desc in work['job_desc']:
            pdf.cell(5, 5, "-", 0, 0)
            pdf.multi_cell(0, 6, job_desc)
        pdf.ln(3)
    pdf.ln(3)

    pdf.add_section_title("Education")
    for education in json_output['education']:
        pdf.add_section_subtitle(f"{education['title']} | {education['institute']}")
        pdf.add_text(f"Location: {education['location']}")
        pdf.add_text(f"Duration: {education['start_date']} - {education['end_date']}")
        pdf.add_text(f"Description: {education['description']}")
    pdf.ln(3)

    pdf.add_section_title("Languages")
    pdf.add_text(', '.join(json_output['languages']))
    pdf.ln(3)

    pdf.add_section_title("Skills")
    pdf.add_text(', '.join(json_output['skills']))
    pdf.ln(3)

    pdf.add_section_title("Certifications")
    for cert in json_output['certification']:
        pdf.add_section_subtitle(f"{cert['title']} | {cert['issuer']}")
        pdf.add_text(f"Issued date: {cert['start_date']} - {cert['expiration_date']}")

    summary_pdf_name = f"{file_name[:-4]}_summary.pdf"
    temp_pdf_path = '/tmp/' + summary_pdf_name
    pdf.output(temp_pdf_path)
    upload_pdf_to_gcs(temp_pdf_path, summary_pdf_name)
    os.remove(temp_pdf_path)
    return get_gcs_file_url(summary_pdf_name)

@app.route('/', methods=['POST'])
def main():
    logging.info("Request received")
    request_json = request.get_json(silent=True)
    if not request_json:
        logging.error("No JSON received in request")
        return 'Invalid request', 400
    if 'original_cv_path' not in request_json or 'cv_id' not in request_json:
        logging.error("No file_name in request JSON")
        return 'Invalid request', 400
    
    try:
        original_cv_path = request_json['original_cv_path']
        cv_id = request_json['cv_id']
        file_name = urllib.parse.unquote(original_cv_path.split('/')[-1])
        bucket_name = 'your-bucket-name'
        pdf_bytes = download_pdf_from_storage(bucket_name, file_name)
        extracted_text = extract_data_from_pdf(pdf_bytes)
        print(extracted_text)
        clean_text = preprocessing(extracted_text)
        json_output = summarize_cv(clean_text)
        if json_output is None:
            logging.error("Failed to summarize CV")
            return 'Failed to process CV', 500
        pdf_url = summarize_to_pdf(json_output, file_name)
        logging.info("PDF summary created successfully")
        json_output['cv_id'] = cv_id
        return json.dumps({"summarized_cv_path": pdf_url, "candidate_cv_data": json_output}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logging.error("Error processing request: %s", e)
        err = f"Error processing request: {e}"
        return err, 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)