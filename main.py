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

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = None

# Function to evaluate FHIRPath expression
def evaluate_fhirpath(data, fhirpath_expression):
    try:
        result = evaluate(data, fhirpath_expression, [])
        return result
    except Exception as e:
        st.error(f"Error in evaluating FHIRPath expression: {str(e)}")
        return None


# Function to convert natural language to FHIRPath query
def generate_fhirpath_query(question):
    few_shot_examples = """
    You are an assistant that converts natural language questions about patient data into FHIRPath queries.
    Here are some examples:

    1. Question: "How old is the patient?"
       FHIRPath Query: entry.resource.where(resourceType = 'Patient').birthDate

    2. Question: "What is the patient's drug allergy?"
       FHIRPath Query: entry.resource.where(resourceType = 'AllergyIntolerance' and category = 'medication').code.text

    3. Question: "What are the patient's current medications?"
       FHIRPath Query: entry.resource.where(resourceType = 'MedicationStatement' and status = 'active').medicationCodeableConcept.text

    4. Question: "What is the patient's gender?"
       FHIRPath Query: entry.resource.where(resourceType = 'Patient').gender

    5. Question: "What is the patient's primary diagnosis?"
       FHIRPath Query: entry.resource.where(resourceType = 'Condition' and clinicalStatus.coding.code = 'active').code.coding.display

    Please convert the following question into a FHIRPath query.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": few_shot_examples},
                {"role": "user", "content": f"Question: {question}\nFHIRPath Query:"},
            ],
            max_tokens=100,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in generating FHIRPath query: {str(e)}"

# Function to convert FHIRPath results to natural language
def convert_to_natural_language(fhir_result):
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
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": few_shot_examples},
                {"role": "user", "content": f"{fhir_result}"},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in converting to natural language: {str(e)}"

# Function to handle ChatGPT interaction
def ask_chatgpt(question, context=""):
    try:
        messages = [
            {"role": "system", "content": "You are a medical assistant answering questions based on provided patient data and general knowledge."},
        ]
        if context:
            messages.append({"role": "assistant", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": question})
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with ChatGPT: {str(e)}"


# Streamlit App
st.title("FHIR-Chat Integration")

# File upload
uploaded_file = st.file_uploader("Upload FHIR JSON file", type=["json"])
if uploaded_file:
    try:
        st.session_state["patient_data"] = json.load(uploaded_file)
        st.success("File loaded successfully!")
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")


# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask a question about the document or anything else.")
if user_input:
    # Add user's question to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Initialize response
    assistant_response = ""

    # Check if document is loaded and query is related to the document
    if st.session_state["patient_data"]:
        # Convert the user's question to FHIRPath
        fhirpath_query = generate_fhirpath_query(user_input)

        # Evaluate the FHIRPath query
        fhir_result = evaluate_fhirpath(st.session_state["patient_data"], fhirpath_query)

        if fhir_result and not isinstance(fhir_result, str) and fhir_result != []:  # Valid result
            # Convert FHIR result to natural language
            assistant_response = convert_to_natural_language(fhir_result)
        else:
            assistant_response = "The requested information is not present in the uploaded document."
    else:
        assistant_response = "No document is loaded. Please upload a JSON document first."

    # If assistant response is empty or needs additional context, use GPT-4
    if "The requested information is not present" in assistant_response:
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
        assistant_response = ask_chatgpt(user_input, context)

    # Add assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Limit chat history to the most recent 20 messages
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]
