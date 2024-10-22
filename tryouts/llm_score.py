import numpy as np
import pandas as pd
import os

import google.generativeai as genai

import os


gemini_api_key = os.getenv('GENAI_API_KEY_CVAI')

genai.configure(api_key=gemini_api_key)

def query_api(model_name, jd, resume):
    if model_name == "gemini":
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        prompt = f"""I'm picking top resume from the collection of resumes given the jd. Given this jd = [{jd}] and the resume = [{resume}], based on your score i'll order them. How well does the candidate's technical skills and other required match the job requirements? Give me a match score between this jd and the resume. The return element should just be a score between 0 and 1, no other text."""
        response = model.generate_content(prompt)

        return response.text
    elif model_name == "OpenAI":
        pass
    else:
        pass