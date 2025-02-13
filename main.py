from fastapi import FastAPI

import uvicorn

import os

from repository.firebase.Firebase import Firebase

from controller import HttpController

from fastapi.middleware.cors import CORSMiddleware

from service.LLMScore  import LLMScore

Firebase.initialize()

LLMScore.initialiseModel()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(HttpController.router)

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

