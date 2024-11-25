import streamlit as st
from fhirpathpy import evaluate
from dotenv import load_dotenv
import json
import openai
from openai import OpenAI
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Function to evaluate FHIRPath expression
def evaluate_fhirpath(data, fhirpath_expression):
    try:
        result = evaluate(data, fhirpath_expression, [])
        return result
    except Exception as e:
        st.error(f"Error in evaluating FHIRPath expression: {str(e)}")
        return None

# Function to handle ChatGPT interaction
def ask_chatgpt(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with ChatGPT: {str(e)}"

# Streamlit App
#st.title("DocuChat")
#st.write("Upload a FHIR JSON file, run FHIRPath queries, and interact with ChatGPT.")

# Sidebar for predefined queries
with st.sidebar:
    st.header("Common Emergency Questions")
    predefined_queries = {
        "How old is the patient?": "entry.resource.where(resourceType = 'Patient').birthDate",
        "What is the patient's drug allergy?": "entry.resource.where(resourceType = 'AllergyIntolerance' and category = 'medication').code.text",
        "What are the patient's current medications?": "entry.resource.where(resourceType = 'MedicationStatement' and status = 'active').medicationCodeableConcept.text",
        "What is the patient's gender?": "entry.resource.where(resourceType = 'Patient').gender",
        "What is the patient's contact phone number?": "entry.resource.where(resourceType = 'Patient').telecom.where(system = 'phone').value",
    }
    query_selection = st.selectbox("Emergency Questions", list(predefined_queries.keys()))
    #custom_query = st.text_input("Or enter a custom FHIRPath query")
    custom_query = []
    if st.button("Get answer to your queries", key="query_button"):
        if "patient_data" in st.session_state:
            query_to_run = predefined_queries.get(query_selection) if not custom_query else custom_query
            result = evaluate_fhirpath(st.session_state["patient_data"], query_to_run)
            if result:
                st.success(f"Result: {result}")
            else:
                st.error("No result found.")
        else:
            st.error("Please upload a FHIR JSON file first.")

# Main content: File upload
st.title("DocuChat")

# Upload files using Streamlit's file uploader
#uploaded_files = st.file_uploader("Upload your documents (PDF, TXT, JSON/FHIR, IPS)", type=["pdf", "txt", "json"], accept_multiple_files=True)
uploaded_files = st.file_uploader("Upload FHIR JSON file", type=["json"])
# Text area for raw text input
raw_text = st.text_area("Or enter your raw text here:", height=150)
if uploaded_files:
    try:
        st.session_state["patient_data"] = json.load(uploaded_files)
        st.success("File loaded successfully!")
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")

# Function to convert FHIRPath results to natural language
def convert_to_natural_language(fhir_result):
    # Few-shot examples to help the model understand the conversion
    few_shot_examples = """
    Examples of converting FHIRPath results to natural language:
    1. FHIRPath Result: "entry.resource.where(resourceType = 'Patient').birthDate: 1987-09-25"
       Natural Language: "The patient's birth date is September 25th, 1987."

    2. FHIRPath Result: "entry.resource.where(resourceType = 'AllergyIntolerance' and category = 'medication').code.text: Penicillin"
       Natural Language: "The patient is allergic to Penicillin."

    3. FHIRPath Result: "entry.resource.where(resourceType = 'Patient').gender: male"
       Natural Language: "The patient is a male."

    Please convert the following FHIRPath result into natural language.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": few_shot_examples},
            {"role": "user", "content": f"{fhir_result}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content
 


# ChatGPT Integration
st.subheader("Chat with ChatGPT")
chat_input = st.text_input("Continue conversations here!")
if st.button("Ask!", key="chat_button"):
    if chat_input:
        chat_response = ask_chatgpt(chat_input)
        st.write(f"ChatGPT Response: {chat_response}")
    else:
        st.warning("Please enter a question to ask ChatGPT.")
