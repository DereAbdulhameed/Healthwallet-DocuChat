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

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize parsed data storage
if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = {}

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
                # Extract the actual value, assuming fhir_result is a list of extracted items
                extracted_value = fhir_result[0] if isinstance(fhir_result, list) and len(fhir_result) > 0 else fhir_result
                natural_language_response = convert_to_natural_language(extracted_value)
                emergency_faqs[question] = natural_language_response
            else:
                emergency_faqs[question] = "Not Found"
        except Exception as e:
            st.error(f"Error processing FAQ '{question}': {str(e)}")
            emergency_faqs[question] = "Error in processing"

            
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

# Function to convert FHIRPath results to natural language
def convert_to_natural_language(fhir_result):
    # Few-shot examples to help the model understand the conversion
    few_shot_examples = """
    Examples of converting FHIRPath results to natural language:
    1. FHIRPath Result: "Patient.birthDate: 1987-09-25"
       Natural Language: "The patient's birth date is September 25th, 1987."

    2. FHIRPath Result: "AllergyIntolerance.substance: Penicillin"
       Natural Language: "The patient is allergic to Penicillin."

    3. FHIRPath Result: "Observation.valueQuantity.value: 120, Observation.valueQuantity.unit: mmHg"
       Natural Language: "The patient's blood pressure reading is 120 mmHg."

    4. FHIRPath Result: "Patient.gender: male"
       Natural Language: "The patient is male."

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
 

# Upload files using Streamlit's file uploader
uploaded_files = st.file_uploader("Upload your documents (PDF, TXT, JSON/FHIR, IPS)", type=["pdf", "txt", "json"], accept_multiple_files=True)

# Text area for raw text input
raw_text = st.text_area("Or enter your raw text here:", height=150)


# Function to parse FHIR/IPS JSON document and store key information in session state
def parse_and_store_fhir_data(data):
    if "entry" in data and isinstance(data["entry"], list):
        for entry in data["entry"]:
            if isinstance(entry, dict) and "resource" in entry:
                resource = entry["resource"]
                resource_type = resource.get("resourceType")

                if resource_type:
                    # Store resources based on type and extract important details
                    if resource_type not in st.session_state.parsed_data:
                        st.session_state.parsed_data[resource_type] = []

                    parsed_resource = extract_key_details(resource)
                    st.session_state.parsed_data[resource_type].append(parsed_resource)
    else:
        st.error("Uploaded JSON file is not in the expected FHIR format with an 'entry' key.")

# Function to extract key details from resources
def extract_key_details(resource):
    resource_type = resource.get("resourceType")

    if resource_type == "MedicationRequest":
        return {
            "resourceType": resource_type,
            "status": resource.get("status"),
            "medication": resource.get("medicationCodeableConcept", {}).get("text"),
            "requester": resource.get("requester", {}).get("display"),
            "authoredOn": resource.get("authoredOn"),
        }

    elif resource_type == "Condition":
        return {
            "resourceType": resource_type,
            "clinicalStatus": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
            "condition": resource.get("code", {}).get("text"),
            "verificationStatus": resource.get("verificationStatus", {}).get("coding", [{}])[0].get("code"),
            "onsetDateTime": resource.get("onsetDateTime"),
        }

    elif resource_type == "AllergyIntolerance":
        return {
            "resourceType": resource_type,
            "status": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
            "allergy": resource.get("code", {}).get("text"),
            "category": resource.get("category", [])[0] if resource.get("category") else "Unknown",
            "criticality": resource.get("criticality"),
        }

    elif resource_type == "Observation":
        return {
            "resourceType": resource_type,
            "category": resource.get("category", [{}])[0].get("coding", [{}])[0].get("display", "Unknown Category"),
            "code": resource.get("code", {}).get("coding", [{}])[0].get("display", "Unknown Code"),
            "status": resource.get("status", "Unknown Status"),
            "effectiveDateTime": resource.get("effectiveDateTime", "Unknown Date"),
            "value": resource.get("valueQuantity", {}).get("value", resource.get("valueCodeableConcept", {}).get("text", "No Value")),
        }

    elif resource_type == "Procedure":
        return {
            "resourceType": resource_type,
            "status": resource.get("status"),
            "procedure": resource.get("code", {}).get("text"),
            "performedDateTime": resource.get("performedDateTime"),
        }

    elif resource_type == "FamilyMemberHistory":
        return {
            "resourceType": resource_type,
            "relationship": resource.get("relationship", {}).get("text"),
            "condition": [
                {
                    "condition": condition.get("code", {}).get("text"),
                    "outcome": condition.get("outcome", {}).get("text"),
                }
                for condition in resource.get("condition", [])
            ],
        }

    # Generic fallback for any other resource types
    else:
        return extract_dynamic_details(resource)


# Function to dynamically extract all available key-value pairs
def extract_dynamic_details(resource):
    extracted_details = {"resourceType": resource.get("resourceType", "Unknown Resource Type")}
    for key, value in resource.items():
        if isinstance(value, dict):
            extracted_details[key] = extract_dynamic_details(value)  # Recursively extract details from nested dicts
        elif isinstance(value, list):
            extracted_details[key] = [
                extract_dynamic_details(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            extracted_details[key] = value
    return extracted_details

## Function to convert parsed information to natural language
#def convert_parsed_data_to_natural_language(resource_type, resources):
#    if resource_type == "MedicationRequest":
#        return [f"The patient was prescribed {resource['medication']} on {resource['authoredOn']} by {resource['requester']}. Status: {resource['status']}."
#                for resource in resources]

#    elif resource_type == "Condition":
#        return [f"The patient has {resource['condition']} with status '{resource['clinicalStatus']}'. It was recorded on {resource['onsetDateTime']}."
#                for resource in resources]

#    return []

def convert_parsed_data_to_natural_language(resource_type, resources):
    important_fields = {
        "MedicationRequest": ["medication", "status", "requester", "authoredOn"],
        "Condition": ["condition", "clinicalStatus", "onsetDateTime"],
        "AllergyIntolerance": ["substance", "status", "criticality"],
        "Patient": ["name", "birthDate", "gender", "address", "telecom"]
    }

    # Check if the resource type has any important fields defined
    if resource_type in important_fields:
        fields_to_include = important_fields[resource_type]
    else:
        # If no specific fields are defined, include everything
        fields_to_include = resources[0].keys() if resources else []

    response_parts = []

    # Iterate over resources and collect the response details
    for resource in resources:
        parts = []
        for field in fields_to_include:
            if field in resource:
                value = resource[field]

                # Convert value to a readable form if it's nested
                if isinstance(value, dict):
                    value = ", ".join([f"{k}: {v}" for k, v in value.items()])
                elif isinstance(value, list):
                    value = ", ".join([str(v) for v in value])

                parts.append(f"{field.replace('_', ' ').capitalize()}: {value}")

        if parts:
            response_parts.append("\n".join(parts))  # Join all parts with a newline for each field

    # Combine all parts into a single response for the resource type
    if response_parts:
        response = f"The patient's {resource_type} information includes:\n\n" + "\n\n".join(response_parts)
        return [response]

    return []



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

# Process the uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:

        if uploaded_file.type == "application/json":
            # Load and parse JSON file if it's a FHIR/IPS file
            try:
                data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                st.write("JSON data successfully loaded.")
                parse_and_store_fhir_data(data)
                #observations = get_observations(data)

                # Display parsed data summary
                st.success("FHIR data parsed and stored successfully.")

                # Populate emergency FAQs immediately after loading the document
                populate_emergency_faqs(data)
                
            except json.JSONDecodeError:
                st.error("Failed to parse JSON. Please check that the file is a valid JSON file.")


# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get the user's question
user_prompt = st.chat_input("Ask anything")

if user_prompt:
    # Validate the user's question
    def validate_question(question):
        if len(question) > 200:
            st.error("Your question is too long. Please keep it concise.")
            return False
        return True

    if validate_question(user_prompt):
        # Add the user's question to the chat history and display it
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        st.chat_message("user").markdown(user_prompt)

        # Initialize variables to hold the response and context data
        response = ""
        context_data = ""

        # Extract relevant data based on user's query and generate response
        if "medication" in user_prompt.lower() or "drug" in user_prompt.lower():
            medication_data = st.session_state.parsed_data.get("MedicationRequest", [])
            if medication_data:
                response = convert_parsed_data_to_natural_language("MedicationRequest", medication_data)
                context_data = "\n\n".join(response)

        elif "condition" in user_prompt.lower() or "health status" in user_prompt.lower():
            condition_data = st.session_state.parsed_data.get("Condition", [])
            if condition_data:
                response = convert_parsed_data_to_natural_language("Condition", condition_data)
                context_data = "\n\n".join(response)

        elif "allergy" in user_prompt.lower() or "drug intolerance" in user_prompt.lower():
            allergy_data = st.session_state.parsed_data.get("AllergyIntolerance", [])
            if allergy_data:
                response = convert_parsed_data_to_natural_language("AllergyIntolerance", allergy_data)
                context_data = "\n\n".join(response)

        # Initialize the assistant response
        assistant_response = ""

        # If the parsed data contains relevant information
        if response:
            assistant_response = "\n".join(response)
        else:
            # Use GPT-4 for dynamic questions and reasoning if no direct response is available
            gpt_prompt = f"""
            You are a medical assistant. The user has asked the following question: "{user_prompt}"

            Based on the following medical data, please answer the user's question as accurately as possible.

            Medical Data:
            {context_data if context_data else "No specific medical data available."}

            Note: If the user asks about interactions or safety for pregnancy, please include that analysis in your response.
            """

            # Include the chat history to provide context
            messages = [{"role": "system", "content": "You are a helpful medical assistant."}]
            for message in st.session_state.chat_history:
                messages.append(message)

            # Add the user's current question and the generated prompt
            messages.append({"role": "user", "content": gpt_prompt})

            # Call GPT-4 to generate a dynamic response
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=300,
                temperature=0.5
            )

            assistant_response = gpt_response.choices[0].message.content

        # Default response if no relevant data is found and GPT cannot provide an answer
        if not assistant_response:
            assistant_response = "I'm unable to find the specific information you requested in the current data. Please provide more details or clarify your question."

        # Add assistant's response to the chat history and display it
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Limit chat history to the most recent 20 messages
        MAX_CHAT_HISTORY = 20
        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
