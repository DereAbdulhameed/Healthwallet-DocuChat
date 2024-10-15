import streamlit as st
import openai
from openai import OpenAI
from brain import get_index_for_documents
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from raga_llm_hub import RagaLLMEval, get_data
import os
import pandas as pd

# Set the title for the Streamlit app
st.title("DocuChat with Evaluation")

# Set up the OpenAI client
client = OpenAI()
load_dotenv()  # Load variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to create vector database from different file types
@st.cache_resource
def create_vectordb(files, filenames, raw_texts):
    # Show a spinner while creating the vectordb
    with st.spinner("Creating vector database..."):
        vectordb = get_index_for_documents(
            [file.getvalue() for file in files if file.type == "application/pdf"],
            filenames,
            [raw_text for raw_text in raw_texts.splitlines() if raw_text.strip()],
            openai.api_key
        )
    return vectordb

# Upload files using Streamlit's file uploader
uploaded_files = st.file_uploader("Upload your documents (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="hidden")

# Text area for raw text input
raw_text = st.text_area("Or enter your raw text here:", height=150)

# If files are uploaded or raw text is provided, create the vectordb and store it in the session state
if uploaded_files or raw_text:
    file_names = [file.name for file in uploaded_files] if uploaded_files else []
    st.session_state["vectordb"] = create_vectordb(uploaded_files, file_names, raw_text)

# Modify the prompt template to let the LLM use its own knowledge if needed
prompt_template = """
    You are a helpful Assistant who answers users' questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence is the context of the document extract with metadata, whenever available. 
    
    Carefully focus on the metadata, especially 'filename' and 'page' whenever answering using the document content.
    
    Make sure to add the filename and page number at the end of the sentence you are citing from the document.

    If the information is not available in the given context, use your general knowledge to answer the question accurately.

    Reply "Not applicable" only if the question is completely irrelevant or outside your capabilities.

    The document content is:
    {doc_extract}
"""

# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("You need to provide a PDF, TXT file, or raw text.")
            st.stop()

    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=3)
    if search_results:
        # If there are results, use them in the response
        doc_extract = "\n".join([result.page_content for result in search_results])
    else:
        # If no relevant document context is found, leave it empty to allow LLM to use its general knowledge
        doc_extract = ""

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

    # Store the response and reference text for evaluation
    reference_text = doc_extract if doc_extract else "No document reference available."
    generated_response = result

    # Evaluation section
    evaluator = RagaLLMEval(api_keys={"OPENAI_API_KEY": openai.api_key})

    # Harmless Evaluation
    evaluator.add_test(
        test_names=["harmless_test"],
        data={
            "prompt": [question],
            "response": [generated_response],
        }
    )

    # Bias Evaluation
    #evaluator.add_test(
    #    test_names="bias_test",
    #    data={
    #        "prompt": question,
    #        "response": generated_response,
    #    },
    #    arguments={"model": "gpt-4", "threshold": 0.5}
    #)

    # Consistency Evaluation
    evaluator.add_test(
        test_names=["consistency_test"],
        data={"prompt": question, "response": generated_response},
        arguments={"threshold": 0.5}
    )

    # Hallucination Evaluation
    evaluator.add_test(
        test_names="hallucination_test",
        data={
            "prompt": question,
            "response": generated_response,
            "context": [reference_text],
        },
        arguments={"model": "gpt-4", "threshold": 0.6}
    )

    # Contextual Relevancy Evaluation
    evaluator.add_test(
        test_names=["contextual_relevancy_test"],
        data={
            "prompt": question,
            "response": generated_response,
            "context": [reference_text],
        },
        arguments={"model": "gpt-4", "threshold": 0.6}
    )

    # Run the evaluations and display the results
    #evaluator.run()
    #st.write("Evaluation Results:")
    #evaluator.print_results()
    

    # Run the evaluations and capture the results
    results = evaluator.run()

    # Instead of printing the results to the terminal, display them in the Streamlit app
    st.write("Evaluation Results:")

    # Assuming `evaluator.get_results()` provides the results in a displayable format
    if hasattr(evaluator, 'get_results'):
        st.write(evaluator.get_results())  # Replace this with the method that returns evaluation results 
    else:
        st.write("No results available to display.")
