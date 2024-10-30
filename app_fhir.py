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

# Function to load FHIR/IPS document
def load_fhir_document(file):
    return json.loads(file.getvalue().decode("utf-8"))

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

# Create the vector database and store it in the session state if files are uploaded
if uploaded_files or raw_text:
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

# Emergency FAQs
faqs = {
    "What should I do if someone is choking?": (
        "1. Encourage them to cough if they can.\n"
        "2. If they can’t breathe, give 5 back blows between the shoulder blades.\n"
        "3. If unsuccessful, follow with 5 abdominal thrusts (Heimlich maneuver).\n"
        "4. Call emergency services if the person continues choking."
    ),
    "How do I perform CPR?": (
        "1. Ensure the person is lying on their back on a firm surface.\n"
        "2. Place the heel of your hand on the center of the chest and interlock your fingers.\n"
        "3. Push down hard and fast (about 100-120 compressions per minute) until help arrives.\n"
        "4. If trained, alternate 30 compressions with 2 rescue breaths."
    ),
    "What should I do in case of a severe allergic reaction?": (
        "1. If available, administer an epinephrine auto-injector (EpiPen) immediately.\n"
        "2. Call emergency services right away.\n"
        "3. Keep the person calm and lying down if possible, and monitor their breathing.\n"
        "4. If they stop breathing, start CPR if trained."
    ),
    "How do I treat a burn?": (
        "1. Cool the burn under running water for at least 10 minutes.\n"
        "2. Cover the burn with a sterile, non-fluffy dressing or cloth.\n"
        "3. Avoid applying any creams or oily substances.\n"
        "4. Seek medical help if the burn is large, deep, or on sensitive areas (face, hands, feet, etc.)."
    ),
    "What should I do if someone is having a heart attack?": (
        "1. Call emergency services immediately.\n"
        "2. Help the person to sit down and stay calm.\n"
        "3. If they are not allergic, give them aspirin (325 mg) to chew slowly.\n"
        "4. Be prepared to start CPR if they lose consciousness."
    ),
    "How can I help someone who is having a seizure?": (
        "1. Stay calm and time the seizure.\n"
        "2. Clear the area of any sharp or harmful objects.\n"
        "3. Place something soft under their head.\n"
        "4. Do not restrain them or put anything in their mouth.\n"
        "5. Call emergency services if the seizure lasts more than 5 minutes."
    )
}


# Display FAQs in the sidebar
st.sidebar.header("Emergency FAQs")
st.sidebar.write("Here are some commonly asked questions about handling emergency situations:")
for question, answer in faqs.items():
    with st.sidebar.expander(question):
        st.write(answer)
