import streamlit as st
import openai
from openai import OpenAI
from brain import get_index_for_documents
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import json
from fhirpathpy import evaluate  # Using fhirpathpy for FHIRPath evaluations

# Set the title for the Streamlit app
st.title("DocuChat")

# Set up the OpenAI client
client = OpenAI()
load_dotenv()  # Load variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to load FHIR/IPS document
def load_fhir_document(file):
    return json.loads(file.getvalue().decode("utf-8"))

# Function to evaluate FHIRPath queries using fhirpathpy
def evaluate_fhirpath(data, fhirpath_expression):
    try:
        # Evaluating FHIRPath expression using fhirpathpy
        result = evaluate(data, fhirpath_expression, [])
        return result
    except Exception as e:
        st.error(f"Error in evaluating FHIRPath expression: {str(e)}")
        return None

# Function to generate FHIRPath queries using GPT
def generate_fhirpath_query(question):
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"Convert the following question into a FHIRPath query: {question}",
        max_tokens=50
    )
    return response.choices[0].text.strip()

# Function to convert FHIRPath results to natural language using GPT
def convert_to_natural_language(fhir_result):
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"Convert the following data into natural language: {fhir_result}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to create vector database from different file types
@st.cache_resource
def create_vectordb(files, filenames, raw_texts):
    with st.spinner("Creating vector database..."):
        vectordb = get_index_for_documents(
            [file.getvalue() for file in files if file.type == "application/pdf"],
            filenames,
            [raw_text for raw_text in raw_texts.splitlines() if raw_text.strip()],
            openai.api_key
        )
    return vectordb

# Upload files using Streamlit's file uploader
uploaded_files = st.file_uploader("Upload your documents (PDF, TXT, JSON/FHIR, IPS)", type=["pdf", "txt", "json"], accept_multiple_files=True, label_visibility="hidden")

# Text area for raw text input
raw_text = st.text_area("Or enter your raw text here:", height=150)

# If files are uploaded or raw text is provided, create the vectordb and store it in the session state
if uploaded_files or raw_text:
    file_names = [file.name for file in uploaded_files] if uploaded_files else []
    st.session_state["vectordb"] = create_vectordb(uploaded_files, file_names, raw_text)

# Define the template for the chatbot prompt
prompt_template = """
    You are a helpful Assistant who answers users' questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence is the context of the document extract with metadata. 
    
    Carefully focus on the metadata, especially 'filename' and 'page' whenever answering.
    
    Make sure to add filename and page number at the end of the sentence you are citing to.

    Also be able to use your general knowledge to give an adequate summary based on the document extract given to you, but do not hallucinate.
        
    Reply "Not applicable" if text is irrelevant.
     
    The document content is:
    {doc_extract}
"""

# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("You need to provide a PDF, TXT file, FHIR, or IPS document.")
            st.stop()

    doc_extract = None

    # If a JSON/FHIR/IPS file is uploaded, handle FHIRPath query
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/json":
            fhir_data = load_fhir_document(uploaded_file)
            fhirpath_query = generate_fhirpath_query(question)
            fhirpath_result = evaluate_fhirpath(fhir_data, fhirpath_query)
            if fhirpath_result:
                doc_extract = convert_to_natural_language(fhirpath_result)
            break

    # If handling PDF or TXT files, search the vectordb for similar content
    if not doc_extract and vectordb:
        search_results = vectordb.similarity_search(question, k=3)
        doc_extract = "\n".join([result.page_content for result in search_results])

    if doc_extract:
        # Update the prompt with the document extract
        prompt[0] = {
            "role": "system",
            "content": prompt_template.format(doc_extract=doc_extract),
        }

        # Add the user's question to the prompt and display it
        prompt.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Display an empty assistant message while waiting for the response
        with st.chat_message("assistant"):
            botmsg = st.empty()

        # Call ChatGPT with streaming and display the response as it comes
        response = []
        result = ""
        for chunk in client.chat.completions.create(
            model="gpt-3.5-turbo", messages=prompt, stream=True
        ):
            text = chunk.choices[0].delta.content
            if text is not None:
                response.append(text)
                result = "".join(response).strip()
                botmsg.write(result)

        # Add the assistant's response to the prompt
        prompt.append({"role": "assistant", "content": result})

        # Store the updated prompt in the session state
        st.session_state["prompt"] = prompt
    else:
        with st.chat_message("assistant"):
            st.write("No relevant data found in the document.")
