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

# Load environment variables
client = OpenAI()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title of the Streamlit page
st.title("DocuChat")

# Few-shot examples for generating FHIRPath queries
few_shot_examples = """
Examples of natural language questions and their corresponding FHIRPath queries:
1. Question: "How old is the patient?"
   FHIRPath Query: Patient.birthDate

2. Question: "What is the patient's drug allergy?"
   FHIRPath Query: AllergyIntolerance.where(category = 'medication').code.text

Please convert the following question into a FHIRPath query.
"""

# Function to generate FHIRPath queries using GPT with few-shot prompting
def generate_fhirpath_query(question):
    response = client.chat.completions.create(
        model="gpt-4",
        prompt=f"{few_shot_examples}\nQuestion: {question}\nFHIRPath Query:",
        max_tokens=50,
        temperature=0.2
    )
    return response.choices[0].text.strip()

# Function to load and parse FHIR/IPS JSON document for observations
def get_observations(data):
    # Check if data contains an "entry" key, indicating it may be in the expected FHIR structure
    entries = data.get("entry") if isinstance(data, dict) else None

    # If the "entry" key exists and holds a list of resources, process it
    if isinstance(entries, list):
        observations = []
        for entry in entries:
            if isinstance(entry, dict) and "resource" in entry:
                resource = entry["resource"]
                
                # Extract details from each observation
                category = resource.get("category", [{}])[0].get("coding", [{}])[0].get("display", "Unknown Category")
                code = resource.get("code", {}).get("coding", [{}])[0].get("display", "Unknown Code")
                status = resource.get("status", "Unknown Status")
                effective_date = resource.get("effectiveDateTime", "Unknown Date")
                result_text = resource.get("valueCodeableConcept", {}).get("text", "No Result Text")
                
                # Append extracted details to the list
                observations.append({
                    "Category": category,
                    "Code": code,
                    "Status": status,
                    "Effective Date": effective_date,
                    "Result": result_text
                })
        return observations
    else:
        st.error("Uploaded JSON file is not in the expected FHIR format with an 'entry' key.")

# Function to evaluate FHIRPath queries using fhirpathpy
def evaluate_fhirpath(data, fhirpath_expression):
    try:
        result = evaluate(data, fhirpath_expression, [])
        return result
    except Exception as e:
        st.error(f"Error in evaluating FHIRPath expression: {str(e)}")
        return None

# Function to convert FHIRPath results to natural language using GPT
def convert_to_natural_language(fhir_result):
    response = client.chat.completions.create(
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
uploaded_files = st.file_uploader("Upload your documents (PDF, TXT, JSON/FHIR, IPS)", type=["pdf", "txt", "json"], accept_multiple_files=True)

# Text area for raw text input
raw_text = st.text_area("Or enter your raw text here:", height=150)

# Process the uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/json":
            # Load and parse JSON file if it's a FHIR/IPS file
            try:
                data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                observations = get_observations(data)
                
                # Display observations extracted from JSON
                if observations:
                    st.write("Patient Observations from JSON:")
                    for obs in observations:
                        st.write(f"- **Category**: {obs['Category']}")
                        st.write(f"  **Code**: {obs['Code']}")
                        st.write(f"  **Status**: {obs['Status']}")
                        st.write(f"  **Effective Date**: {obs['Effective Date']}")
                        st.write(f"  **Result**: {obs['Result']}")
                        st.write("---")
            except json.JSONDecodeError:
                st.error("Failed to parse JSON. Please check that the file is a valid JSON file.")
        else:
            # Process non-JSON files for vector database creation
            file_names = [file.name for file in uploaded_files] if uploaded_files else []
            st.session_state["vectordb"] = create_vectordb(uploaded_files, file_names, raw_text)

# Define the chatbot template
prompt_template = """
    You are a helpful Assistant who answers users' questions based on multiple contexts given to you.

    The evidence is the context of the document extract with metadata. 
    
    Carefully focus on the metadata, especially 'filename' and 'page' whenever answering.
    
    Make sure to add filename and page number at the end of the sentence you are citing to.

    Focus on the context of the document extract with metadata.
    Add filename and page number if citing directly from a document.
    Use general knowledge if document-based information is irrelevant.
    
    The document content is:
    {doc_extract}
"""

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get the user's question
user_prompt = st.chat_input("Ask anything")

if user_prompt:
    # Add the user's question to the chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    st.chat_message("user").markdown(user_prompt)

    # Retrieve the vector database and search for document-based answers
    vectordb = st.session_state.get("vectordb", None)
    doc_extract = None

    if vectordb:
        # Search vectordb for relevant content if available
        search_results = vectordb.similarity_search(user_prompt, k=3)
        doc_extract = "\n".join([result.page_content for result in search_results])

    # Generate a response based on the document or switch to chat history
    if doc_extract:
        # Document-based response with context
        prompt = [{"role": "system", "content": prompt_template.format(doc_extract=doc_extract)}]
        prompt.append({"role": "user", "content": user_prompt})
    else:
        # Fallback to general chat history if no document content found
        prompt = [{"role": "system", "content": "You are a helpful assistant"}]
        prompt.extend(st.session_state.chat_history)

    # Get the response from the assistant
    response = client.chat.completions.create(
        model="gpt-4",
        messages=prompt
    )

    assistant_response = response.choices[0].message.content

    # Add assistant's response to the chat history and display it
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
