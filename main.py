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
        print(f"Error loading JSON file: {str(e)}")
        exit(1)

    # List of FHIRPath queries
    queries = [
        ("How old is the patient?", "entry.resource.where(resourceType = 'Patient').birthDate"),
        ("What is the patient's drug allergy?", "entry.resource.where(resourceType = 'AllergyIntolerance' and category = 'medication').code.text"),
        ("What are the patient's current medications?", "entry.resource.where(resourceType = 'MedicationStatement' and status = 'active').medicationCodeableConcept.text"),
        ("What is the patient's gender?", "entry.resource.where(resourceType = 'Patient').gender"),
        ("Does the patient have a history of smoking?", "entry.resource.where(resourceType = 'Observation' and code.coding.display = 'Tobacco smoking status').valueCodeableConcept.text"),
        ("What is the patient's primary diagnosis?", "entry.resource.where(resourceType = 'Condition' and clinicalStatus.coding.code = 'active').code.coding.display"),
        ("What is the patient's contact phone number?", "entry.resource.where(resourceType = 'Patient').telecom.where(system = 'phone').value"),
        ("What are the patient's laboratory test results?", "entry.resource.where(resourceType = 'Observation' and category.coding.code = 'laboratory').valueQuantity.value"),
        ("What is the patient's blood type?", "entry.resource.where(resourceType = 'Observation' and code.coding.display = 'Blood group').valueCodeableConcept.text"),
        ("What is the patient's weight?", "entry.resource.where(resourceType = 'Observation' and code.coding.display = 'Body Weight').valueQuantity.value"),
    ]

    # Iterate through queries and evaluate them
    for question, fhirpath_expression in queries:
        print(f"Question: {question}")
        result = evaluate_fhirpath(patient_data, fhirpath_expression)
        if result:
            print(f"Result: {result}")
        else:
            st.error("Please upload a FHIR JSON file first.")

# Main content: File upload
st.title("FHIR Data Explorer")
uploaded_file = st.file_uploader("Upload FHIR JSON file", type=["json"])
if uploaded_file:
    try:
        st.session_state["patient_data"] = json.load(uploaded_file)
        st.success("File loaded successfully!")
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")

# ChatGPT Integration
st.subheader("Chat with ChatGPT")
chat_input = st.text_input("Continue conversations here!")
if st.button("Ask!", key="chat_button"):
    if chat_input:
        chat_response = ask_chatgpt(chat_input)
        st.write(f"ChatGPT Response: {chat_response}")
    else:
        st.warning("Please enter a question to ask ChatGPT.")
