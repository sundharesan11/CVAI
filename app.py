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
import os
# from tryouts.embedding import preprocess, generate_bert_embedding, tokenizer, model
from tryouts.embedding import preprocess, process_text, model
from tryouts.summarizer import summarize
from datetime import timedelta
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
st.markdown('<h1 class="title">CV Matcher AI</h1>', unsafe_allow_html=True)


def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

custom_css = load_css("utilities/styless.css")

st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


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
    
    expiration_time = timedelta(hours=1)
    files = []
    for blob in blobs:
        if blob.name.endswith("/"):
            continue 
        
        signed_url = blob.generate_signed_url(expiration=expiration_time)
        files.append((blob.name.split("/")[-1], signed_url))
    return files

def list_files_raw(folder):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=folder)
    return [(blob.name.split("/")[-1], blob.public_url) for blob in blobs if blob.name != folder]

def download_textfile(bucket_name, file_name):
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(file_name)
    try:
        file_data = blob.download_as_bytes()
        return file_data.decode('utf-8')
        # file_data = blob.download_as_bytes()
        # return io.BytesIO(file_data).read().decode('utf-8')
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
    text = txt_file.getvalue() 
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    return text
    
# def extract_text_from_txt(txt_file):
#     try:
#     # Ensure the file isn't empty
#         st.write(f"Uploaded file size: {txt_file.size} bytes")  # Display file size

#         text = txt_file.getvalue() 

#         if isinstance(text, bytes):
#             st.write(f"Raw content (binary): {text[:100]}") 
#             text = text.decode('utf-8')
#         return text
#     except Exception as e:
#         st.error(f"Error extracting text: {e}")
#     return ""


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


def jd_embedding(job_description):
    job_description_text = preprocess(job_description)
    job_description_embedding = model.encode(job_description_text, convert_to_numpy=True)
    return job_description_embedding


def resume_embedding(df):
    df['preprocessed_content'] = df['Content'].apply(preprocess)
    df['resume_embedding'] = df['preprocessed_content'].apply(lambda x: model.encode(x, convert_to_numpy=True))
    resume_embeddings = np.vstack(df['resume_embedding'].values)
    embedding_dim = resume_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)  
    index.add(resume_embeddings) 
    return index



def main():
    db = firestore.client()

        
    st.subheader("Upload Files")


    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div data-testid = "column">', unsafe_allow_html=True)
        st.subheader('Drop CVs here')
        uploaded_resumes = st.file_uploader("**Upload Resumes**", type=["pdf", "png", "jpg", "jpeg", "txt"], key="resume", accept_multiple_files=True)

        if uploaded_resumes is not None:
            for uploaded_resume in uploaded_resumes:
                if uploaded_resume is not None:
                    folder_name = 'CVs'
                    with st.spinner('Uploading Resume...'):
                        file_url = upload(uploaded_resume, folder_name)
                        st.success(f"Resume uploaded to Firebase Storage: {file_url}")
                        text = extract_text_from_pdf(uploaded_resume)
                        text =  process_text(text)
                        summary = summarize(text)
                        text_file_url = upload_text(text, 'processed_resume', uploaded_resume.name.replace('.pdf', '.txt'))
                        text_file_url2 = upload_text(summary, 'summary_resume', uploaded_resume.name.replace('.pdf', '.txt'))

                        st.success(f"Extracted text from resumes uploaded to Firebase Storage: {text_file_url}")  
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:    
        st.markdown('<div data-testid="column">', unsafe_allow_html=True)
        st.subheader('Drop JDs here')      
        uploaded_jds = st.file_uploader("**Upload Job Description**", type=["pdf", "png", "jpg", "jpeg", "txt"], key="jd", accept_multiple_files=True)

        if uploaded_jds is not None:
            for uploaded_jd in uploaded_jds:
                if uploaded_jd is not None:
                    folder_name = 'JobDescription'
                    with st.spinner('Uploading Job Description...'):
                        file_url = upload(uploaded_jd, folder_name)
                        st.success(f"Job Description uploaded to Firebase Storage: {file_url}")
                        ext = os.path.splitext(file_url)[-1]
                        print("ext:", ext)
                        if ext == '.pdf':
                            text = extract_text_from_pdf(uploaded_jd)
                        if ext == '.txt':
                            text = extract_text_from_txt(uploaded_jd)
                        jd_summary = summarize(text)
                        text_file_url = upload_text(text, 'processed_jd', uploaded_jd.name.replace('.pdf', '.txt'))
                        text_file_url2 = upload_text(jd_summary, 'summary_jd', uploaded_jd.name.replace('.pdf', '.txt'))
                        st.success(f"Extracted text from jd uploaded to Firebase Storage: {text_file_url}")  
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("Available Files")

    # col3, col4 = st.columns(2)
    
    # with col3:
    #     st.markdown('<div data-testid="column">', unsafe_allow_html=True)
    #     st.subheader("Uploaded Job Descriptions")
    #     jds = list_files('JobDescription')
    #     if jds:
    #         for name, url in jds:
    #             st.markdown(f"[{name}]({url})", unsafe_allow_html=True)
    #     else:
    #         st.write("No job descriptions found.")
    
    #     st.markdown('</div>', unsafe_allow_html=True)

    # with col4:
    #     st.markdown('<div data-testid="column">', unsafe_allow_html=True)
    #     with st.container():
    #         st.subheader("Uploaded Resumes")
    #         resumes = list_files('CVs')
    #         if resumes:
    #             for name, url in resumes:
    #                 st.markdown(f"[{name}]({url})", unsafe_allow_html=True)
    #         else:
    #             st.write("No resumes found.")
    #     st.markdown('</div>', unsafe_allow_html=True)

    top_n = st.number_input('Number of Top matching CVs to pick from available CVs', min_value=1, max_value=100, value=5)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Uploaded JDs")
        jds = list_files('JobDescription')
        if jds:            
            clicked_file = None 
            for name, url in jds:
                col1, col2 = st.columns([3, 1])
                with col1:
                    file_name = os.path.splitext(name)[0]
                    f2 = download_textfile("cvai-92a44.appspot.com", "summary_jd" + "/" + file_name + ".txt")
                    f2 = f2[:150] + "..."
                    button_label = f"""{file_name}\n 
\u00A0 \n
{f2}"""
                    if st.button(button_label, key=f"file_{name}"): 
                        clicked_file = name  
                        # st.write(f"You selected: {clicked_file}")
                with col2:
                    st.markdown(f"[Open JD]({url})")     
                st.markdown("<hr>", unsafe_allow_html=True)
            if clicked_file:
                with col6:
                    st.subheader(f"Top {top_n} resumes matching the selected Job Description")
                    data = []
                    resumes = list_files('CVs')
                    for resume_file, url in resumes:
                        print("")
                        resume_file = str(resume_file)
                        resume_file = resume_file.split(",")[0].strip("()' ")
                        resume_file = os.path.splitext(resume_file)[0]
                        resume_file = resume_file + ".txt"
                        print("Name:", resume_file)
                        content = download_textfile("cvai-92a44.appspot.com", "processed_resume" + "/" + resume_file)
                        data.append({'Filename': resume_file.split('/')[-1].rsplit('.', 1)[0], 'Content': content, 'URL': url})

                    df = pd.DataFrame(data)

                    csv_file_path = 'processed_resumes.csv'
                    df.to_csv(csv_file_path, index=False)
                    
                    index = resume_embedding(df)
                    clicked_file = os.path.splitext(clicked_file)[0]
                    clicked_file = clicked_file + ".txt"
                    print("Clicked_file_nme:",clicked_file)
                    jd_content = download_textfile('cvai-92a44.appspot.com', "processed_jd" + "/" + clicked_file)
                    jd_embed = jd_embedding(jd_content)
                    
                    distances, indices = index.search(np.array([jd_embed]), top_n)
                    top_resumes = df.iloc[indices[0]].reset_index(drop=True)
                    for idx, a in top_resumes.iterrows():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            file_name = a['Filename']
                            f2 = download_textfile("cvai-92a44.appspot.com", "summary_resume" + "/" + file_name + ".txt")
                            f2 = f2[:150] + "..."
                            button_label = f"""{file_name}\n 
\u00A0 \n
{f2}"""
                            st.button(button_label)
                        with col2:
                            st.markdown(f"[Open CV]({a['URL']})") 
                        st.markdown("<hr>", unsafe_allow_html=True)

        else:
            st.write("No jds found.")
    with col6:
        if not clicked_file:
            resumes = list_files('CVs')

            if resumes:
                st.subheader("Uploaded CVs")
                for name, url in resumes:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        file_name = os.path.splitext(name)[0]
                        f2 = download_textfile("cvai-92a44.appspot.com", "summary_resume" + "/" + file_name + ".txt")
                        f2 = f2[:150] + "..."
                        button_label = f"""{file_name}\n 
\u00A0 \n
{f2}"""
                        st.button(button_label)
                    with col2:
                        st.markdown(f"[Open CV]({url})") 
                    st.markdown("<hr>", unsafe_allow_html=True)




if __name__ == "__main__":
    main()
