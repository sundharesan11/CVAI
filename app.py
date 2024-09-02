import firebase_admin
from firebase_admin import credentials, storage, firestore
import streamlit as st

st.set_page_config(layout="wide")

def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("git .css")

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


def main():
    initialize_firebase()
    db = firestore.client()
    
    st.markdown('<h1 class="title">CVAI Application</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.7, 0.3])

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
        
        if st.checkbox("View Uploaded Resumes"):
            resumes = list_files('CVs')
            if resumes:
                for name, url in resumes:
                    st.markdown(f"[{name}]({url})", unsafe_allow_html=True)    
            else:
                st.write("No resumes found.")
        
        if st.checkbox("View Uploaded Job Descriptions"):
            jds = list_files('JobDescription')
            if jds:
                for name, url in jds:
                    st.markdown(f"[{name}]({url})", unsafe_allow_html=True)
            else:
                st.write("No job descriptions found.")
        


if __name__ == "__main__":
    main()
