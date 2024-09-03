import firebase_admin
from firebase_admin import credentials, storage, firestore
import streamlit as st
import os
import requests
from io import BytesIO
import fitz
import urllib.parse
import PyPDF2



st.set_page_config(layout="wide")

def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("utilities\styles.css")

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("utilities/cvai-92a44-firebase-adminsdk-ne70g-d566124d0d.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'cvai-92a44.appspot.com'
        })


def upload(file, folder):
    try:
        bucket = storage.bucket()
        if file is not None:
            file_name = file.name
            file_path = f"{folder}/{file_name}"
            blob = bucket.blob(file_path)
            blob.upload_from_string(file.read(), content_type=file.type)
            print(f"Uploaded file to {file_path}")
            return blob.public_url
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None
    
def list_files(folder):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=folder)
    return [(blob.name.split("/")[-1], blob.public_url) for blob in blobs if blob.name != folder]

def download_file(url, file_name):
    url = urllib.parse.quote(url, safe=':/')
    
    response = requests.get(url)
    if response.status_code == 200:

        file_path = os.path.join("/tmp", file_name)
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        raise Exception(f"Failed to download file from {url}. Status code: {response.status_code}")


def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text



def download_all_blobs_in_folder(bucket_name, folder_name):
    storage_client = storage.bucket(bucket_name)
    blobs = storage_client.list_blobs(prefix=folder_name)

    tmp_dir = os.path.join(os.getcwd(), 'tmp', folder_name)
    os.makedirs(tmp_dir, exist_ok=True)
    for blob in blobs:
        file_name = os.path.basename(blob.name)
        if file_name:
            file_path = os.path.join(tmp_dir, file_name)
            blob.download_to_filename(file_path)
            print(f"Downloaded {file_name} to {file_path}")


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def save_text_to_file(file_name, text, folder_name):

    tmp_dir = os.path.join(os.getcwd(), 'tmptxt', folder_name)
    os.makedirs(tmp_dir, exist_ok=True)
    txt_file_name = file_name.replace('.pdf', '.txt')
    txt_file_path = os.path.join(tmp_dir, txt_file_name)
    
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)
    
    print(f"Saved extracted text to {txt_file_path}")

def process_and_save_all_pdfs(folder_name):
    tmp_dir = os.path.join(os.getcwd(), 'tmp', folder_name)
    
    for file_name in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, file_name)
        
        if file_name.lower().endswith('.pdf'):
            print(f"Extracting text from {file_name}...")
            text = extract_text_from_pdf(file_path)
            save_text_to_file(file_name, text, folder_name)


def main():
    initialize_firebase()
    db = firestore.client()
    
    st.markdown('<h1 class="title">CVAI Application</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.5, 0.3, 0.2])

    with col1:

        st.header("Upload Files")
        st.subheader("Upload Resume")
        uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "png", "jpg", "jpeg", "txt"], key="resume")

        st.subheader("Upload Job Description")
        uploaded_jd = st.file_uploader("Upload Job Description", type=["pdf", "png", "jpg", "jpeg", "txt"], key="jd")
        
        if uploaded_resume is not None:
            folder_name = 'CVs'
            with st.spinner('Uploading Resume...'):
                file_url = upload(uploaded_resume, folder_name)
                st.success(f"Resume uploaded to Firebase Storage: {file_url}")

        if uploaded_jd is not None:
            folder_name = 'JobDescription'
            with st.spinner('Uploading Job Description...'):
                file_url = upload(uploaded_jd, folder_name)
                st.success(f"Job Description uploaded to Firebase Storage: {file_url}")
                

    with col2:

        st.header("View Uploaded Files")
        
        if st.button("View Uploaded Resumes"):
            resumes = list_files('CVs')
            if resumes:
                for name, url in resumes:
                    st.markdown(f"[{name}]({url})", unsafe_allow_html=True)    
            else:
                st.write("No resumes found.")
        
        if st.button("View Uploaded Job Descriptions"):
            jds = list_files('JobDescription')
            if jds:
                for name, url in jds:
                    st.markdown(f"[{name}]({url})", unsafe_allow_html=True)
            else:
                st.write("No job descriptions found.")
        

    with col3:
        if st.button("Download and Process Resumes"):
            try:
                bucket_name = 'cvai-92a44.appspot.com'
                folder_name = 'CVs/'
                download_all_blobs_in_folder(bucket_name, folder_name)
                process_and_save_all_pdfs(folder_name)
                st.success(f"Resumes are downloaded and processed")
            except Exception as e:
                print(f"Error: {e}")

        if st.button("Download and Process JDs"):
            try:
                bucket_name = 'cvai-92a44.appspot.com'
                folder_name = 'JobDescription/'
                download_all_blobs_in_folder(bucket_name, folder_name)
                process_and_save_all_pdfs(folder_name)
                st.success(f"JDs are downloaded and processed")
            except Exception as e:
                print(f"Error: {e}")



if __name__ == "__main__":
    main()
