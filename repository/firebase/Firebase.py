from firebase_admin import credentials, initialize_app, storage # type: ignore
from firebase_admin import _apps

from fastapi import HTTPException

from repository.firebase.serviceAccountKey import service_account_key_file

import os

from dotenv import load_dotenv

from datetime import timedelta

load_dotenv()

class Firebase:
    
    def initialize():

        if not _apps:
            cred = credentials.Certificate(service_account_key_file())
            initialize_app(cred, {
                'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET"),
                'databaseURL': os.getenv("FIREBASE_DATABASE_URL")
            })
        else:
            print("Firebase is already initialized.")
    
    async def uploadFile(file,folderName):

        file.file.seek(0)

        blob = storage.bucket().blob(f"{folderName}/{file.filename}")

        try:
            blob.upload_from_file(file.file, content_type=file.content_type)
            blob.make_public() 

            return blob.public_url
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to upload file {file.filename}: {str(e)}"
            )
    
    async def uploadText(text, folderName, filename):

        blob = storage.bucket().blob(f"{folderName}/{filename}")

        try:
        
            blob.upload_from_string(text, content_type="text/plain")
            blob.make_public()
            
            return blob.public_url
        
        except Exception as e:
             
            raise Exception(f"Failed to upload text: {str(e)}")
        
    async def listFiles(folder):

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
    
    async def downloadTextfile(bucket_name, file_name):

        bucket = storage.bucket(bucket_name)
        blob = bucket.blob(file_name)

        try:
            file_data = blob.download_as_bytes()
            return file_data.decode('utf-8')
        
        except Exception as e:
            raise Exception(f"Error downloading file {file_name}: {e}")