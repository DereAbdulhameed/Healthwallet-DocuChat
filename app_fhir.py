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
import requests

# Load environment variables
client = OpenAI()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
FHIR_SERVER_URL = os.getenv("FHIR_SERVER_URL")

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title of the Streamlit page
st.title("DocuChat")

# Few-shot prompting
few_shot_examples = """
Examples of natural language questions and their corresponding FHIRPath queries:
1. Question: "How old is the patient?"
   FHIRPath Query: Patient.birthDate

2. Question: "What is the patient's drug allergy?"
   FHIRPath Query: AllergyIntolerance.where(category = 'medication').code.text

3. Question: "What are the patient's current medications?"
   FHIRPath Query: MedicationStatement.where(status = 'active').medicationCodeableConcept.text

4. Question: "What is the patient's gender?"
   FHIRPath Query: Patient.gender

5. Question: "Does the patient have a history of smoking?"
   FHIRPath Query: Observation.where(code.coding.display = 'Tobacco smoking status').valueCodeableConcept.text

6. Question: "What is the patient's primary diagnosis?"
   FHIRPath Query: Condition.where(clinicalStatus = 'active').code.coding.display

7. Question: "What is the patient's contact phone number?"
   FHIRPath Query: Patient.telecom.where(system = 'phone').value

8. Question: "What are the patient's laboratory test results?"
   FHIRPath Query: Observation.where(category.coding.code = 'laboratory').valueQuantity.value

9. Question: "What is the patient's blood type?"
   FHIRPath Query: Observation.where(code.coding.display = 'Blood group').valueCodeableConcept.text

10. Question: "What is the patient's weight?"
    FHIRPath Query: Observation.where(code.coding.display = 'Body Weight').valueQuantity.value

Please convert the following question into a FHIRPath query.
"""

# Function to generate FHIRPath queries using GPT with few-shot prompting
def generate_fhirpath_query(question):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": few_shot_examples},
            {"role": "user", "content": f"Question: {question}\nFHIRPath Query:"}
        ],
        max_tokens=50,
        temperature=0.2
    )
    return response.choices[0].message.content


# Function to query the FHIR server using FHIRPath
def query_fhir_server(fhir_query):
    try:
        headers = {'Content-Type': 'application/fhir+json'}
        response = requests.get(f"{FHIR_SERVER_URL}/{fhir_query}", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying FHIR server: {str(e)}")
        return None

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
        messages=[
            {"role": "system", "content": "Convert the following data into natural language."},
            {"role": "user", "content": f"{fhir_result}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

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
                st.write("JSON data successfully loaded.")

                # Populate emergency FAQs immediately after loading the document
                populate_emergency_faqs(data)

            except json.JSONDecodeError:
                st.error("Failed to parse JSON. Please check that the file is a valid JSON file.")

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

    # Generate FHIRPath query for user's question
    fhir_query = generate_fhirpath_query(user_prompt)
    fhir_data = query_fhir_server(fhir_query)

    # If FHIR data is found, convert it to natural language
    if fhir_data:
        natural_language_response = convert_to_natural_language(fhir_data)
        assistant_response = natural_language_response
    else:
        # Fallback to general GPT-4 response
        messages = [{"role": "system", "content": "You are a helpful assistant"}]
        messages.extend(st.session_state.chat_history)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        assistant_response = response.choices[0].message.content

    # Add assistant's response to the chat history and display it
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

        
# Emergency FAQs to answer upon document upload
emergency_faqs = {
    "What are the patient's vital signs?": "Observation.where(category.coding.code = 'vital-signs').valueQuantity.value",
    "Does the patient have any known chronic medical conditions?": "Condition.where(clinicalStatus = 'active').code.coding.display",
    "Is the patient allergic to any medications?": "AllergyIntolerance.where(category = 'medication').code.text",
    "Does the patient have a history of surgeries?": "Procedure.where(status = 'completed').code.coding.display",
    "What is the patient's current medication list?": "MedicationStatement.where(status = 'active').medicationCodeableConcept.text",
    "Is there any known family medical history that is relevant?": "FamilyMemberHistory.condition.code.coding.display",
    "Does the patient smoke or use alcohol?": "Observation.where(code.coding.display = 'Tobacco smoking status').valueCodeableConcept.text",
    "Has the patient traveled recently?": "Observation.where(code.coding.display = 'Travel history').valueString"
}

# Function to populate FAQs from document data
def populate_emergency_faqs(data):
    for question, fhirpath_expression in emergency_faqs.items():
        try:
            fhir_result = evaluate(data, fhirpath_expression, [])
            if fhir_result:
                natural_language_response = convert_to_natural_language(fhir_result)
                emergency_faqs[question] = natural_language_response
            else:
                emergency_faqs[question] = "Not Found"
        except Exception as e:
            st.error(f"Error processing FAQ '{question}': {str(e)}")
            emergency_faqs[question] = "Error in processing"


# Display updated Emergency FAQs in the sidebar
st.sidebar.header("Emergency FAQs (Patient Information)")
st.sidebar.write("These are key questions a doctor would ask upon first contact.")
for question, answer in emergency_faqs.items():
    with st.sidebar.expander(question):
        st.write(answer if answer != "Not Found" else "Not Found")