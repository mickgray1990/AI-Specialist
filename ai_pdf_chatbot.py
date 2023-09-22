import streamlit as st
import openai
import pandas as pd
import numpy as np
import PyPDF2
from multiprocessing import Pool, cpu_count
import os
import re
import shutil
import sys
from pathlib import Path
import time


EMBEDDING_MODEL = "text-embedding-ada-002"
RETRIES = 5
n_cpu_cores = 1

#system_content = """Using the information from the provided context, please answer the question at the end of the prompt. When referencing information from the provided context, use in-text citations formatted in the Vancouver referencing style (e.g., [1], [2], [3]).
#
#If the required information to answer the question is not present in the provided context, kindly state that the information is not available in the provided sources and do not attempt to generate an answer without adequate reference.
#
#After your response, please provide a reference list citing the sources used in your answer, with each reference listed on a new line.
#
#It is critical that your answer to the question is generated from the provided context. DO NOT make up an answer. If the answer is not within the provided context, just say so.
#"""

system_content = """Using the information from the provided context, please answer the question at the end of the prompt. When referencing information from the provided context, use in-text citations formatted in the Vancouver referencing style (e.g., [Source 1], [Source 2], [Source 3]).

If the required information to answer the question is not present in the provided context, kindly state that the information is not available in the provided sources and do not attempt to generate an answer without adequate reference.

It is critical that your answer to the question is generated from the provided context. DO NOT make up an answer. If the answer is not within the provided context, just say so.


"""


def prompt_generator(contexts, filenames, question):
    combined_context = "\n\n".join([f"Reference for Source {i+1}: '{filenames[i]}'\nProvided Context for Source {i+1}: '{context}'" for i, context in enumerate(contexts)])
    prompt = f'''

    Provided Context: 
    
    {combined_context}

    Question: {question}

    {system_content}

    '''

    print(prompt)

    return prompt

# List of folder names
folders = ["Docs", "Docs Database", "Embeddings"]

# Function to create a folder if it doesn't exist
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Loop through the list of folders and create them if they don't exist
for folder in folders:
    create_folder(folder)

def load_embeddings_csv_to_np(file_path):
    df = pd.read_csv(file_path)
    # Load FileName column as well
    filenames = df['FileName'].values
    texts = df['Text'].values
    num_cols = len(df.columns) - 2  # Subtract 2 to account for the 'Text' and 'FileName' column
    embeddings = np.array([df[f'Text Chunk {i}'].values for i in range(num_cols)]).T
    # Return filenames along with texts and embeddings
    return filenames, texts, embeddings

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]

def cosine_similarity_chunk(args):
    query_embeddings, context_embeddings_chunk = args
    dot_product = np.dot(context_embeddings_chunk, query_embeddings)
    query_norm = np.linalg.norm(query_embeddings)
    context_norms = np.linalg.norm(context_embeddings_chunk, axis=1)
    cosine_similarities = dot_product / (query_norm * context_norms)
    return cosine_similarities

def cosine_similarity_parallel(query_embeddings, context_embeddings):
    n_cores = min(n_cpu_cores, cpu_count())
    context_chunks = np.array_split(context_embeddings, n_cores)

    with Pool(n_cores) as p:
        results = p.map(cosine_similarity_chunk, [(query_embeddings, chunk) for chunk in context_chunks])

    cosine_similarities = np.concatenate(results)
    return cosine_similarities

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        text = " ".join(page.extract_text() for page in pdf.pages)
    return text

def text_to_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    return text

def import_pdf_from_folder(folder_path):
    if not os.path.exists(folder_path):
        st.error(f'The folder {folder_path} does not exist')
        return None

    # Filter PDF files with a case-insensitive check
    pdf_files = [file for file in os.listdir(folder_path) if re.match(r'.*\.pdf$', file, re.IGNORECASE)]

    if not pdf_files:
        st.error(f'The folder {folder_path} does not contain any PDF files')
        return None

    total_files = len(pdf_files)
    st.write(f'Total number of PDF files: {total_files}')

    text_chunks = []
    for index, file in enumerate(pdf_files, start=1):
        st.write(f'Processing file {index} of {total_files} - "{file}"')
        text = extract_text_from_pdf(os.path.join(folder_path, file))
        text = clean_text(text)
        chunks = text_to_chunks(text, 1000, 200)
        text_chunks.extend((chunk, file) for chunk in chunks)
    return np.array(text_chunks)

def save_embeddings_to_csv(np_array, texts, filenames, file_path, file_mode='w'):
    columns = [f'Text Chunk {i}' for i in range(np_array.shape[1])]
    df = pd.DataFrame(np_array, columns=columns)
    df['Text'] = texts
    df['FileName'] = filenames
    if file_mode == 'a' and os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
        print("Embeddings have been appended to the existing file.")
    else:
        df.to_csv(file_path, index=False)
        print("Embeddings have been written to a new file.")

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    for i in range(RETRIES):
        try:
            result = openai.Embedding.create(
                model=model,
                input=text
            )
            return result["data"][0]["embedding"]
        except openai.error.APIError as e:
            if i < RETRIES - 1:
                time.sleep(2 ** i)
                continue
            else:
                raise
        except Exception as e:
            if i < RETRIES - 1:
                time.sleep(2 ** i)
                continue
            else:
                raise

def run_embedding_generator(api_key, filename, pdf_text_chunks, append, should_delete_existing_file=False):
    try:
        openai.api_key = api_key
        
        if pdf_text_chunks is None:
            return

        texts = np.array([chunk[0] for chunk in pdf_text_chunks])
        filenames = np.array([chunk[1] for chunk in pdf_text_chunks])
        context_embeddings = np.empty((0, 1536))
        total_texts = len(texts)
        
        for i, text in enumerate(texts):
            context_embeddings = np.vstack((context_embeddings, get_embedding(text=text)))
            progress_percent = ((i + 1) / total_texts) * 100
            st.info(f"Generating Embeddings - {progress_percent:.2f}%")
        
        file_mode = 'a' if append else 'w'
        file_path = os.path.join("Embeddings", filename)
        
        if should_delete_existing_file and os.path.exists(file_path):
            os.remove(file_path)
            st.info(f"Deleted the existing file: {filename}")
        
        save_embeddings_to_csv(context_embeddings, texts, filenames, file_path, file_mode)
        st.info(f"Embeddings have been successfully saved as {file_path}!")
        
        dir_path = os.path.join("Docs Database", os.path.splitext(os.path.basename(file_path))[0])
        os.makedirs(dir_path, exist_ok=True)
        source_dir = "Docs"
        for filename in os.listdir(source_dir):
            source_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(dir_path, filename)
            if os.path.exists(destination_file):
                os.remove(destination_file)  # remove the existing file in the destination
            shutil.move(source_file, destination_file)
        
        st.info(f"Docs have been moved to {dir_path}.")
        st.success("Embeddings generator executed successfully. Please refresh the page and proceed in 'Use Existing Embeddings' mode.")
    except Exception as e:
        st.error(f"Unexpected error occurred: {e}")


def display_and_upload_pdf_files():
    docs_path = 'Docs'
    if os.path.exists(docs_path):
        files = os.listdir(docs_path)
        if files:
            st.write('List of files in "Docs" folder:')
            st.write(files)
        else:
            st.write('No files found in "Docs" folder.')
    else:
        st.error('"Docs" folder does not exist.')

    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_path, uploaded_file.name)
            with open(file_path, 'wb') as out_file:
                out_file.write(uploaded_file.getvalue())
            st.write(f"Uploaded {uploaded_file.name} to {docs_path}.")

def generate_new_embeddings(api_key):
    filename = st.text_input("Enter a filename for the embeddings:")
    if filename:
        filename += ".csv"
        st.write(f"Filename Selected: {filename}")
        file_path = os.path.join("Embeddings", filename)
        append_option = 'No'
        delete_option = False
        if os.path.exists(file_path):
            append_option = st.radio("The file already exists. Do you want to append to it?", ('Yes', 'No'))
            if append_option == 'No':
                delete_option = st.checkbox("Warning: Selecting 'No' will permanently delete the current embeddings file. Are you sure?")
                if not delete_option:  # Check if the user does not confirm the deletion
                    st.warning(f"File {filename} will not be deleted.")
                    st.stop()  # Stop execution if the user doesn't confirm the deletion
                else:
                    st.warning(f"File {filename} will be deleted.")
        if st.button("Proceed with " + ("Appending" if append_option == 'Yes' else "Generating New") + " Embeddings"):
            try:
                pdf_text_chunks = import_pdf_from_folder('Docs')
                if pdf_text_chunks is not None and len(pdf_text_chunks) > 0:
                    run_embedding_generator(api_key, filename, pdf_text_chunks, append=(append_option == 'Yes'), should_delete_existing_file=delete_option)
            except Exception as e:
                st.error(f"Error occurred: {e}")


import re

def clean_output(content, most_similar_filenames):
    pattern = r"\[?Source ([\d, ]+)]?"
    source_numbers = []

    for match in re.finditer(pattern, content):
        numbers_str = match.group(1)
        numbers = [int(num) for num in re.findall(r'\d+', numbers_str)]
        source_numbers.extend(numbers)

    mapping_dict = {}
    reference_list = []
    added_filenames = set()

    for number in source_numbers:
        index = number - 1
        if 0 <= index < len(most_similar_filenames):
            filename = most_similar_filenames[index]
            if filename not in added_filenames:
                added_filenames.add(filename)
                mapping_index = len(added_filenames)
                mapping_dict[number] = mapping_index
                reference_list.append(f"{mapping_index}. {filename}")

    content = re.sub(pattern,
                     lambda m: '[' + ', '.join(str(mapping_dict.get(int(num), ''))
                                               for num in re.findall(r'\d+', m.group(1))) + ']'
                     if re.findall(r'\d+', m.group(1)) else '',
                     content)

    content = content.replace("[]", "")
    content += "\n\nReferences:\n" + "\n".join(reference_list)

    return content



def use_existing_embeddings():
    COMPLETIONS_MODEL_OPTIONS = {
        "GPT3.5": "gpt-3.5-turbo",
        "GPT4": "gpt-4"
    }
    
    # Choose GPT version
    chosen_option = st.selectbox("Choose GPT version:", list(COMPLETIONS_MODEL_OPTIONS.keys()))
    COMPLETIONS_MODEL = COMPLETIONS_MODEL_OPTIONS[chosen_option]
    st.write(f"Selected GPT Version: {chosen_option} (API Model: {COMPLETIONS_MODEL})")
    
    if chosen_option == "GPT4":
        st.warning('Warning: GPT4 is expensive, please monitor your usage through OpenAI\'s website.')
    
    # Choose the number of content sources
    n_content_sources = st.number_input("Number of Content Sources:", value=10, step=1)
    st.write(f"Number of Content Sources: {n_content_sources}")
    
    # Now add the code from your 'main()' function here
    embedding_folder = Path('Embeddings')  # Path to your folder
    files = ['Please select a .CSV file'] + [f.name for f in embedding_folder.glob('*.csv') if f.is_file()]
    chosen_file = st.selectbox("Choose a embeddings file", files)
    if chosen_file != 'Please select a .CSV file':
        
        question = st.text_area("Enter your question:", height=50) # Change to text_area for multiline and wrapping
        question = question.replace("\n", " ") # Replace newline characters with spaces
        if st.button('Generate Answer'):
            if question:

                placeholder = st.empty()  # Creating a placeholder for the warning message

                # Load the embeddings
                with st.spinner('Loading...'):
                    filenames, texts, context_embeddings = load_embeddings_csv_to_np(embedding_folder / chosen_file)

                    query_embeddings = get_embedding(text=question, model=EMBEDDING_MODEL)

                    cosine_similarities = cosine_similarity_parallel(query_embeddings, context_embeddings)

                    # Get indices of top most similar texts
                    sorted_indices = np.argsort(cosine_similarities)[::-1]  # Descending order
                    top_indices = sorted_indices[:n_content_sources]

                    # Return the top most similar texts and filenames
                    most_similar_texts = texts[top_indices]
                    most_similar_filenames = filenames[top_indices]

                    # No need to redefine context as 'most_similar_texts' since we're going to use both texts and filenames
                    completion = openai.ChatCompletion.create(
                        model=COMPLETIONS_MODEL,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": prompt_generator(most_similar_texts, most_similar_filenames, question)}
                        ]
                    )

                placeholder.success('Success. Ask another question?')  # Replacing the placeholder warning message
                print(completion.choices[0].message.content)
                # Display the updated content
                st.markdown(clean_output(completion.choices[0].message.content, most_similar_filenames))  # Displaying output in markdown for better text wrapping

                # Add context sources
                for i, (src, fn) in enumerate(zip(most_similar_texts, most_similar_filenames)):
                    with st.expander(f"Source"):
                        st.markdown(f"<b>File Name:<b> {fn}\n\n<b>Chunk Text:<b> {src}", unsafe_allow_html=True)  # Display each source in markdown format for better text wrapping

            else:
                st.warning('Please enter a question.')
    else:
        st.warning('Please select a embeddings file.')



def app():
    st.title("Specialist AI Advisor")
    
    # Add note directly underneath the title
    st.info("This application is currently in beta testing. For reporting bugs and suggesting improvements, please contact harrison.ferrier@outlook.com.  \n\n"
            "Please be aware that usage of the OpenAI API is subject to cost. It is highly recommended to configure your OpenAI API usage limits and monitor your usage regularly.  \n\n"
            "GPT-4 significantly outperforms GPT-3.5; however, be mindful of API usage, as GPT-4 is expensive.")

    api_key_file = "api_key.txt"
    
    # Check if the API key file exists and read the key if it does
    if os.path.exists(api_key_file):
        with open(api_key_file, 'r') as file:
            api_key = file.readline().strip()
        label = "OpenAI API Key:"
    else:
        api_key = ""
        label = "Enter your OpenAI API Key:"
    
    # Display the appropriate message based on the existence of the API key file
    api_key = st.text_input(label, value=api_key, type="password")
    
    if api_key:
        openai.api_key = api_key
        
        # Save API key to the file
        with open(api_key_file, 'w') as file:
            file.write(api_key)
        
        option = st.selectbox(
            "Choose an option:",
            ("Select Configuration", "Upload PDF Files", "Generate New Embeddings", "Use Existing Embeddings", "Delete API Key"),
            index=0
        )
        st.write(f"Selected Option: {option}")
        
        if option == "Upload PDF Files":
            display_and_upload_pdf_files()
        elif option == "Generate New Embeddings":
            generate_new_embeddings(api_key)
        elif option == "Use Existing Embeddings":
            use_existing_embeddings()
        elif option == "Delete API Key":
            if st.button("Confirm Delete"):
                os.remove(api_key_file)
                st.success("API Key Deleted Successfully!")


if __name__ == '__main__':
    app()
