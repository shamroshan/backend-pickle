from fastapi import FastAPI, HTTPException 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
import pickle


load_dotenv()

app = FastAPI()  
 
openai.api_key =  os.environ.get("OPENAI_KEY")
connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vegascg-genie-chat.azurewebsites.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel): 
    role: str
    content: str

@app.get("/")
async def helloapp():
    return {"message": "Hello App"}
 
async def startup_event():
    global index, chat_engine
    # Azure Blob Storage setup
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "indexed-file"
    blob_name = "index.pkl"

    # Download the indexed data from Azure Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob().readall()

    # Deserialize the data (assuming it's stored in a serialized format like pickle)
    indexed_data = pickle.loads(blob_data)

    # Load the data into the VectorStoreIndex
    index = indexed_data
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )
app.add_event_handler("startup", startup_event)

async def enhance_response(response: str, query: str) -> str:
    """Enhance the response by providing additional context or clarification."""
    prompt = (
        f"Improve the following response based on the query:\n\n"
        f"Query: {query}\n"
        f"Response: {response}\n\n"
        """Provide a more detailed and accurate response in markdown format. Use lists, bold texts, italics, bullets, points, etc. to visualize attractively. 
        You are a specialized chatbot designed to answer only technical questions related to VegasCG and QMS Standards for the American Petroleum Institute (API) and the International Organization for Standardization (ISO).
        Adhere to these guidelines:
        1. Only provide answers to questions that are strictly technical and related to VegasCG or QMS Standards for API and ISO.
        2. Provide a clear, concise explanation. The explanation should be appropriate to your answer. Don't display anything like 'Here is an enhanced response'. The answer must be structured and visually attractive.
        3. You are a specialized chatbot designed to answer only technical questions related to VegasCG and QMS Standards for the American Petroleum Institute (API) and the International Organization for Standardization (ISO). 
        4. If a user asks about topics outside your area of expertise, such as general knowledge, politely inform them that you are not defined to provide guidance on those subjects. 
        5. If the query is non-technical, general knowledge or unrelated:
           - answer as "I'm not defined to answer non-technical questions. 
 
        Please make sure to follow these rules strictly."""
    )

    try:
        openai_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert providing enhanced answers based on given queries and responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        enhanced_content = openai_response.choices[0].message.content
        return enhanced_content.strip()
    except Exception as e:
        print(f"Error enhancing response: {e}")
        return response

@app.options("/chat")
async def options_chat_endpoint():
    return {}

@app.post("/chat")
async def chat(message: Message):
    if message.role != "user":
        raise HTTPException(status_code=400, detail="Invalid role")

    # Generate initial response from the chatbot
    response_stream = chat_engine.stream_chat(message.content)
    response_chunks = [chunk for chunk in response_stream.response_gen]
    response = "".join(response_chunks)

    # Enhance the response for better clarity
    enhanced_response = await enhance_response(response, message.content)
    return {"role": "assistant", "content": enhanced_response}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 


