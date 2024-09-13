import firebase_admin
from firebase_admin import credentials, storage, firestore
import streamlit as st
import os
import requests
from io import BytesIO
import fitz
import urllib.parse
import PyPDF2
import io
import pandas as pd
import numpy as np
import faiss
import time
from tryouts.embedding import preprocess, generate_bert_embedding, tokenizer, model
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def initialize_firebase():
    try:
        firebase_credentials = st.secrets["firebase"]["credentials"]
        cred = credentials.Certificate(eval(firebase_credentials))
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'cvai-92a44.appspot.com'
            })
            logger.info("Firebase initialized successfully.")
        else:
            logger.info("Firebase already initialized.")
    except KeyError:
        logger.error("Error: Firebase credentials not found in secrets.")
    except FileNotFoundError:
        logger.error("Error: Service account key file not found. Please check the path.")
    except Exception as e:
        logger.error(f"An error occurred while initializing Firebase: {e}")

initialize_firebase()


st.set_page_config(layout="wide")
st.markdown('<h1 class="title">CVAI Application</h1>', unsafe_allow_html=True)


def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_js(js_file):
    with open(js_file, "r") as f:
        js_code = f.read()
    return js_code
custom_css = load_css("utilities/styles.css")
scroll_js = load_js("utilities/scroll_script.js")

st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
st.markdown(f"<script>{scroll_js}</script>", unsafe_allow_html=True)



# def initialize_firebase():
#     try:
#         if not firebase_admin._apps: 
#             cred = credentials.Certificate("utilities\cvai-92a44-firebase-adminsdk-ne70g-0d2f0c7a8e.json")
#             firebase_admin.initialize_app(cred, {
#                 'storageBucket': 'cvai-92a44.appspot.com'
#             })
#             print("Firebase initialized successfully.")
#         else:
#             print("Firebase already initialized.")
#     except FileNotFoundError:
#         print("Error: Service account key file not found. Please check the path.")
#     except Exception as e:
#         print(f"An error occurred while initializing Firebase: {e}")


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

def download_textfile(bucket_name, file_name):
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(file_name)
    try:
        file_data = blob.download_as_bytes()
        return io.BytesIO(file_data).read().decode('utf-8')
    except Exception as e:
        st.error(f"Error downloading file {file_name}: {e}")
        return ""

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


def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_text_from_txt(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as file:
        text = file.read()
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

def upload_text(text, folder_name, file_name):
    bucket = storage.bucket()
    blob = bucket.blob(f"{folder_name}/{file_name}")
    blob.upload_from_string(text)
    return blob.public_url

def get_latest_jd_file():
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="JobDescription/")
    
    latest_file = max(blobs, key=lambda blob: blob.updated)
    
    return latest_file

def read_blob_as_text(blob):
    return blob.download_as_text()

def jd_embedding():
    latest_jd_blob = get_latest_jd_file()
    job_description = read_blob_as_text(latest_jd_blob)
    st.write("**Given Job Description:**")
    st.write(job_description)
    job_description_text = preprocess(job_description)
    job_description_embedding = generate_bert_embedding(job_description_text, tokenizer, model)
    return job_description_embedding


def resume_embedding(df):
    df['preprocessed_content'] = df['Content'].apply(preprocess)
    df['resume_embedding'] = df['preprocessed_content'].apply(lambda x: generate_bert_embedding(x, tokenizer, model))
    resume_embeddings = np.vstack(df['resume_embedding'].values)
    embedding_dim = resume_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  
    index.add(resume_embeddings) 
    return index



def main():
    
    db = firestore.client()

    st.sidebar.title("Navigation")

    st.sidebar.markdown("""
    <a class="navigation-link" href="#upload-files" onclick="smoothScrollTo('upload-files')">Upload Files</a>
    <br>
    <a class="navigation-link" href="#view-files" onclick="smoothScrollTo('view-files')">View Uploaded Files</a>
    <br>
    <a class="navigation-link" href="#process-files" onclick="smoothScrollTo('process-files')">Process Files</a>
    <br>
    <a class="navigation-link" href="#Get Matching Resumes" onclick="smoothScrollTo('Get Matching Resumes')">Get Matching Resumes</a>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div data-testid='upload-files'>", unsafe_allow_html=True)
        st.header("Upload Files")

        uploaded_resumes = st.file_uploader("**Upload Resumes**", type=["pdf", "png", "jpg", "jpeg", "txt"], key="resume", accept_multiple_files=True)

        uploaded_jd = st.file_uploader("**Upload Job Description**", type=["pdf", "png", "jpg", "jpeg", "txt"], key="jd")


        if uploaded_resumes is not None:
            for uploaded_resume in uploaded_resumes:
                if uploaded_resume is not None:
                    folder_name = 'CVs'
                    with st.spinner('Uploading Resume...'):
                        file_url = upload(uploaded_resume, folder_name)
                        st.success(f"Resume uploaded to Firebase Storage: {file_url}")

                        text = extract_text_from_pdf(uploaded_resume)
                        text_file_url = upload_text(text, 'processed_resume', uploaded_resume.name.replace('.pdf', '.txt'))
                        st.success(f"Extracted text uploaded to Firebase Storage: {text_file_url}")                

        if uploaded_jd is not None:
            folder_name = 'JobDescription'
            with st.spinner('Uploading Job Description...'):
                file_url = upload(uploaded_jd, folder_name)
                st.success(f"Job Description uploaded to Firebase Storage: {file_url}")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div data-testid='view-files'>", unsafe_allow_html=True)
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
        
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div data-testid='process-files'>", unsafe_allow_html=True)
        st.header("Process Files")

        if st.button("Download and Process Resumes"):
            try:
                bucket_name = 'cvai-92a44.appspot.com'
                folder_name = 'CVs/'
                download_all_blobs_in_folder(bucket_name, folder_name)
                folder_name = process_and_save_all_pdfs(folder_name)
                st.success(f"Resumes are downloaded and processed")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Download and Process JDs"):
            try:
                bucket_name = 'cvai-92a44.appspot.com'
                folder_name = 'JobDescription/'
                download_all_blobs_in_folder(bucket_name, folder_name)
                process_and_save_all_pdfs(folder_name)
                st.success(f"JDs are downloaded and processed")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div data-testid='Get Matching Resumes'>", unsafe_allow_html=True)
        st.header("Get Matching Resumes")
        st.markdown("##### Select Resumes to be processed")
        
        if 'selected_files' not in st.session_state:
            st.session_state.selected_files = []

        file_names = list_files('processed_resume')
        
        st.write("Files found:", len(file_names))
        
        if file_names:
            for file_name in file_names:
                file_name_str = str(file_name) 
                clean_file_name = file_name_str.split('/')[-1].strip("()' ")
                clean_file_name = clean_file_name.rsplit('.', 1)[0]

                is_selected = st.checkbox(clean_file_name, key=f"checkbox_{file_name_str}")
                if is_selected:
                    if file_name_str not in st.session_state.selected_files:
                        st.session_state.selected_files.append(file_name_str)
                else:
                    if file_name_str in st.session_state.selected_files:
                        st.session_state.selected_files.remove(file_name_str)
        else:
            st.write("No files available to select.")
                       
        top_n = st.number_input('Number of Top matching resumes', min_value=1, max_value=100, value=5)

        if st.button("Get Matching Resumes"):
            selected_files = st.session_state.selected_files
            if selected_files:
                data = []
                for file_name in selected_files:
                    file_name = file_name.split(",")[0].strip("()' ")
                    content = download_textfile('cvai-92a44.appspot.com', "processed_resume"+"/"+file_name)
                    data.append({'Filename': file_name.split('/')[-1], 'Content': content})

                df = pd.DataFrame(data)

                csv_file_path = 'processed_resumes.csv'
                df.to_csv(csv_file_path, index=False)
                index = resume_embedding(df)
                jd_embed = jd_embedding() 
                distances, indices = index.search(np.array([jd_embed]), top_n)
                top_resumes = df.iloc[indices[0]].reset_index(drop=True)
                st.write(f"**Top {top_n} Matching resumes:**")
                st.write(top_resumes['Filename'])

        st.markdown("</div>", unsafe_allow_html=True)
                


if __name__ == "__main__":
    main()
