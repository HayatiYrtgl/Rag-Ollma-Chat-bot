import os
import shutil
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# load data from directory
def load_document(path: str):
    document_reader = PyPDFDirectoryLoader(path)
    return document_reader.load()



# text splitting
# recrusive function
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)




def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def add_to_chroma(chunks: list[Document], user_id: str):
    # create database for individual usage
    db = Chroma(
        persist_directory=f"dbs_{user_id}", embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database(user_id: str):
    # clear the database when user quit
    if os.path.exists(f"dbs_{user_id}"):
        shutil.rmtree(f"dbs_{user_id}")



def query_rag(query_text: str, user_id: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=f"dbs_{user_id}", embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # create the context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # chat with prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # format the prompt template
    prompt = prompt_template.format(context=context_text, question=query_text)

    # define the llm model (it has to run in background "ollama run mistral")
    model = OllamaLLM(model="mistral")

    # feed the model
    response_text = model.invoke(prompt)

    # get the sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text



def index_all_files():
    # index all files and new files
    print("🔄 Indexing documents...")
    documents = load_document("data")[1:]
    chunks = split_documents(documents)
    add_to_chroma(chunks, user_id="admin")
    print("✅ Indexing complete. Please restart the program!")
    exit()

if __name__ == "__main__":
    # for docker

    # index section
    index_bool = input("Do u want to index files?(y/n):").lower().strip()
    index_bool = True if index_bool=="y" else False
    if index_bool:
        index_all_files()
    print("Type quit for exit the program...")

    # main loop
    while True:

        # get input from user
        question = input("What is your Question? :").lower().strip()

        # quit if user input is quit
        if question=="quit":
            exit()
        # else ask the ai
        else:
            query_rag(query_text=question, user_id="admin")

