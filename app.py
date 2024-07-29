import gradio as gr
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_huggingface import HuggingFacePipeline
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.prompts.prompt import PromptTemplate


from pathlib import Path
import chromadb
from unidecode import unidecode

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import transformers
import torch
import tqdm 
import accelerate
import re

import fitz
from PIL import Image

# extract secret HF token
token = os.environ["HF_TOKEN"]

# create list of LLM model paths
list_llm = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "distilgpt2",
    "HuggingFaceTB/SmolLM-1.7B-Instruct"
]

list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# alternative models (hugging face inference points)
# list_llm = [
#     "mistralai/Mistral-7B-Instruct-v0.2", 
#     "mistralai/Mixtral-8x7B-Instruct-v0.1", \
#     "mistralai/Mistral-7B-Instruct-v0.1", \
#     "google/gemma-7b-it",
#     "google/gemma-2b-it", \
#     "HuggingFaceH4/zephyr-7b-beta", \
#     "HuggingFaceH4/zephyr-7b-gemma-v0.1", \
#     "meta-llama/Llama-2-7b-chat-hf", \
#     "microsoft/phi-2", \
#     "TinyLlama/TinyLlama-1.1B-Chat-v1.0", \
#     "mosaicml/mpt-7b-instruct", \
#     "tiiuae/falcon-7b-instruct", \
#     "google/flan-t5-xxl"
# ]

# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size, chunk_overlap):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    
    for loader in loaders:
        pages.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(
       chunk_size = chunk_size, 
       chunk_overlap = chunk_overlap
    )
    doc_splits = text_splitter.split_documents(pages)
    
    # Clean up unnecessary spaces in each document split
    for i, doc in enumerate(doc_splits):
        cleaned_content = ' '.join(doc.page_content.split())
        doc_splits[i].page_content = cleaned_content
        print(f"Chunk {i+1}: {len(cleaned_content)} characters")
        print(doc.page_content)
    
    return doc_splits


# Create vector database using Chroma
def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
        # persist_directory=default_persist_directory
    )
    
    return vectordb

def load_tokenizer(llm_model):
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    
    return tokenizer 

def load_model(llm_model):
    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        device_map='auto',
        torch_dtype=torch.float32,
        token=True,
        load_in_8bit=False,
    )
    
    return model

def create_pipeline(llm_model, max_tokens, top_k, temperature):
    tok = load_tokenizer(llm_model)
    mod = load_model(llm_model)
    pipe = pipeline(
        task='text-generation',
        model=mod,
        tokenizer=tok,
        max_new_tokens=max_tokens, 
        do_sample=True,
        top_k=top_k,
        trust_remote_code=True,
        num_return_sequences=1, 
        eos_token_id=tok.eos_token_id
    )
    
    pipeline_llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': temperature})
    
    return pipeline_llm

# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, system_prompt, progress=gr.Progress()):
    progress(0.1, desc="Initializing HF tokenizer...")

    
    progress(0.5, desc="Initializing HF Hub...")
    # HuggingFacePipeline uses local model
    # Note: it will download model locally...
    # llm = create_pipeline(llm_model, max_tokens, top_k, temperature)

    # HuggingFaceHub uses HF inference endpoints
    # Use of trust_remote_code as model_kwargs
    # Warning: langchain issue
    # URL: https://github.com/langchain-ai/langchain/issues/6080
    llm = HuggingFaceEndpoint(
        repo_id=llm_model, 
        temperature = temperature,
        max_new_tokens = max_tokens,
        top_k = top_k,
        huggingfacehub_api_token=token
    )
    
    # additional possible models to use:
    # if llm_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
    #     llm = HuggingFaceEndpoint(
    #         repo_id=llm_model, 
    #         # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k, "load_in_8bit": True}
    #         temperature = temperature,
    #         max_new_tokens = max_tokens,
    #         top_k = top_k,
    #         load_in_8bit = True,
    #         token=token
    #     )
    # elif llm_model in ["HuggingFaceH4/zephyr-7b-gemma-v0.1","mosaicml/mpt-7b-instruct"]:
    #     raise gr.Error("LLM model is too large to be loaded automatically on free inference endpoint")
    #     llm = HuggingFaceEndpoint(
    #         repo_id=llm_model, 
    #         temperature = temperature,
    #         max_new_tokens = max_tokens,
    #         top_k = top_k,
    #     )
    # elif llm_model == "microsoft/phi-2":
    #     # raise gr.Error("phi-2 model requires 'trust_remote_code=True', currently not supported by langchain HuggingFaceHub...")
    #     llm = HuggingFaceEndpoint(
    #         repo_id=llm_model, 
    #         # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k, "trust_remote_code": True, "torch_dtype": "auto"}
    #         temperature = temperature,
    #         max_new_tokens = max_tokens,
    #         top_k = top_k,
    #         trust_remote_code = True,
    #         torch_dtype = "auto",
    #     )
    # elif llm_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
    #     llm = HuggingFaceEndpoint(
    #         repo_id=llm_model, 
    #         # model_kwargs={"temperature": temperature, "max_new_tokens": 250, "top_k": top_k}
    #         temperature = temperature,
    #         max_new_tokens = 250,
    #         top_k = top_k,
    #     )
    # elif llm_model == "meta-llama/Llama-2-7b-chat-hf":
    #     raise gr.Error("Llama-2-7b-chat-hf model requires a Pro subscription...")
    #     llm = HuggingFaceEndpoint(
    #         repo_id=llm_model, 
    #         # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k}
    #         temperature = temperature,
    #         max_new_tokens = max_tokens,
    #         top_k = top_k,
    #     )
    # else:
    #     llm = HuggingFaceEndpoint(
    #         repo_id=llm_model, 
    #         # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k, "trust_remote_code": True, "torch_dtype": "auto"}
    #         # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k}
    #         temperature = temperature,
    #         max_new_tokens = max_tokens,
    #         top_k = top_k,
    #         huggingfacehub_api_token=token
    #     )
        
    progress(0.75, desc="Defining buffer memory...")
    
    # general_system_template = r""" 
    # Given a specific context, please give a short answer to the question, covering the required advices in general and then provide the names all of relevant(even if it relates a bit) products. 
    # ----
    # {context}
    # ----
    # """
    # general_user_template = "Question:```{question}```"
    # messages = [
    #             SystemMessagePromptTemplate.from_template(general_system_template),
    #             HumanMessagePromptTemplate.from_template(general_user_template)
    # ]
    # qa_prompt = ChatPromptTemplate.from_messages( messages )
        
    
    # template = """You are named MARLIN a smart agent who offers knowledge about MARIN (Maritime Research Institute Netherlands). 
    # Given a specific context, please give a short and accurate answer to the question. Always finish your answer with a sign-off: MARLIN. 
    # Chat History: {chat_history}
    # Context: {context}
    # Follow Up Input: {question}
    # Helpful Answer: """

    # PROMPT = PromptTemplate(
    #     input_variables=["chat_history", "context", "question"], 
    #     template =template
    # )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    
    progress(0.8, desc="Defining retrieval chain...")
    retriever_db = vector_db.as_retriever(search_type="similarity", search_kwargs={'k': top_k})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
    # qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever = retriever_db,
        chain_type = "stuff", 
        return_source_documents = True,
        verbose = False,
        memory = memory,
        # chain_type_kwargs={"prompt":PROMPT}
        # combine_docs_chain_kwargs={"prompt": PROMPT}
        # return_generated_question=False,
    )
    progress(0.9, desc="Done!")
    
    return qa_chain


# Generate collection name for vector database
#  - Use filepath as input, ensuring unicode text   
def create_collection_name(filepath):
    # Extract filename without extension
    collection_name = Path(filepath).stem
    
    ## Remove space
    collection_name = collection_name.replace(" ","-") 
    
    ## ASCII transliterations of Unicode text
    collection_name = unidecode(collection_name)
    
    ## Remove special characters
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    
    ## Limit length to 50 characters
    collection_name = collection_name[:50]
    
    ## Minimum length of 3 characters
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
        
    ## Enforce start and end as alphanumeric character
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
        
    print('Filepath: ', filepath)
    print('Collection name: ', collection_name)
    
    return collection_name


# Initialize database
def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()):
    # Create list of documents (when valid)
    list_file_path = [x.name for x in list_file_obj if x is not None]
    
    # Create collection_name for vector database
    progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(list_file_path[0])

    # Load document and create splits
    progress(0.25, desc="Loading document...")
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    
    # Create or load vector database (global vector_db)
    progress(0.5, desc="Generating vector database...")
    vector_db = create_db(doc_splits, collection_name)
    
    progress(0.9, desc="Done!")
    
    return vector_db, collection_name, "Complete!"


# Initialize LLM
def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, system_prompt, progress=gr.Progress()):

    llm_name = list_llm[llm_option]    
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, system_prompt, progress)
    
    return qa_chain, "Complete!"


# Format chat history into user and bot messages
def format_chat_history(message, chat_history):
    formatted_chat_history = []
    
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
        
    return formatted_chat_history
    

def conversation(qa_chain, message, history):
    
    # Generate response using QA chain
    formatted_chat_history = format_chat_history(message, history)
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    
    # Extract retriever from the chain
    retriever = qa_chain.retriever
    print(retriever)
    results = retriever.get_relevant_documents(message)

    # Access the documents and similarity scores
    for result in results:
        print('RAG inspect:', result)
        print("Document:", result.page_content)  # Adjust based on your result format
        print("Metadata:", result.metadata)
        # print("Similarity Score:", result.score)  # Adjust based on your result format
        
    # Print response to understand its structure
    print("QA Chain Response:", response)
    
    # Handling response structure
    if "result" in response:
        response_answer = response["result"]
    elif "answer" in response:
        response_answer = response["answer"]
    else:
        raise KeyError("Response does not contain 'result' or 'answer' key")
    
    # Extract helpful answer if necessary
    if "Helpful Answer:" in response_answer:
        response_answer = response_answer.split("Helpful Answer:")[-1]
        
    response_sources = response["source_documents"]
    
    # Handle possible fewer sources
    response_source1 = response_sources[0].page_content.strip() if len(response_sources) > 0 else "No source available"
    # response_source2 = response_sources[1].page_content.strip() if len(response_sources) > 1 else "No source available"
    # response_source3 = response_sources[2].page_content.strip() if len(response_sources) > 2 else "No source available"
    
    response_source1_page = response_sources[0].metadata["page"] + 1 if len(response_sources) > 0 else "N/A"
    # response_source2_page = response_sources[1].metadata["page"] + 1 if len(response_sources) > 1 else "N/A"
    # response_source3_page = response_sources[2].metadata["page"] + 1 if len(response_sources) > 2 else "N/A"
    
    # Print to debug
    print("Response Answer:", response_answer)
    print("Response Sources:", response_sources)
    
    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    
    # Handle possible missing image path
    image_path = None
    if len(response_sources) > 0:
        image_path = render_file(response_sources[0].metadata["source"], response_source1_page)
    print("Image Path:", image_path)    
    
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, image_path


    # formatted_chat_history = format_chat_history(message, history)
    # #print("formatted_chat_history",formatted_chat_history)
   
    # # Generate response using QA chain
    # response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    # # response = qa_chain({"query": message})
    
    # # if response_answer.find("Helpful Answer:") != -1:
    # #     response_answer = response_answer.split("Helpful Answer:")[-1]
    # print(response)
    # response_answer = response["result"]
    # if "Answer:" in response_answer:
    #     response_answer = response_answer.split("Helpful Answer:")[-1]
        
    # response_sources = response["source_documents"]
    # response_source1 = response_sources[0].page_content.strip()
    # response_source2 = response_sources[1].page_content.strip()
    # response_source3 = response_sources[2].page_content.strip()
    
    # # Langchain sources are zero-based
    # response_source1_page = response_sources[0].metadata["page"] + 1
    # response_source2_page = response_sources[1].metadata["page"] + 1
    # response_source3_page = response_sources[2].metadata["page"] + 1
    # # print ('chat response: ', response_answer)
    # # print('DB source', response_sources)
    
    # # Append user message and response to chat history
    # new_history = history + [(message, response_answer)]
    # # new_history =  [(message, response_answer)] # removed memory of past history

    # image_path = render_file(response_sources[0].metadata["source"], response_source1_page)
    # print(image_path)    
    # # return gr.update(value=""), new_history, response_sources[0], response_sources[1] 
    # return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, image_path
    # # return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page, image
    

def clear_conversation(qa_chain):
    qa_chain.memory.clear()
    return qa_chain, gr.update(value=""), [], "", "", None


def upload_file(file_obj):
    list_file_path = []
    
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)

    return list_file_path


def render_file(file_name, file_page):
    doc = fitz.open(file_name)
    page = doc[file_page - 1]
    pix = page.get_pixmap()
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    image_path = f"./page_{file_page}.png"
    image.save(image_path)
    
    return image_path


def demo():
    # set visual properties
    image_path = './marlin_logo.png' # Replace with your image file path
    absolute_path = os.path.abspath(image_path)
    css_properties =  ".gradio-container {background: url('file=marlin_logo.png'); background-repeat: no-repeat; background-size: contain; background-position: center center;}"

    with gr.Blocks(theme="Soft", css=css_properties) as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()
        
        gr.Markdown(
        """<center><h2>MARLIN (PDF-based chatbot)</center></h2>
        <h3>Ask any questions about your PDF documents</h3>""")
        
        with gr.Tab("Step 1 - Upload PDF"):
            with gr.Row():
                document = gr.Files(height=100, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload your PDF documents (single or multiple)")
        

        with gr.Tab("Step 2 - Process document"):
            with gr.Row():
                db_btn = gr.Radio(["ChromaDB"], label="Vector database type", value = "ChromaDB", type="index", info="Choose your vector database")
    
            with gr.Accordion("Advanced options - Document text splitter", open=False):   
                with gr.Row():
                    slider_chunk_size = gr.Slider(minimum = 100, maximum = 1000, value=600, step=10, label="Chunk size", info="Chunk size", interactive=True)
                with gr.Row():
                    slider_chunk_overlap = gr.Slider(minimum = 10, maximum = 200, value=40, step=10, label="Chunk overlap", info="Chunk overlap", interactive=True)
            
            with gr.Row():
                db_progress = gr.Textbox(label="Vector database initialization", value="None")
            
            with gr.Row():
                db_btn = gr.Button("Generate vector database")
            
        with gr.Tab("Step 3 - Choose your LLM model"):
            with gr.Row():
                llm_btn = gr.Radio(list_llm_simple, label="Choose your LLM", value = list_llm_simple[0], type="index", interactive=True)
            
            with gr.Row():
                system_prompt_input = gr.Textbox(label="System Prompt", placeholder="Enter a system prompt here...")
            
            with gr.Accordion("Advanced options - LLM configuration", open=False):   
                with gr.Row():
                    slider_llm_temperature = gr.Slider(minimum = 0.01, maximum = 1.0, value=0.5, step=0.1, label="Temperature", info="Temperature", interactive=True)
                with gr.Row():
                    slider_max_tokens = gr.Slider(minimum = 50, maximum = 4096, value=500, step=10, label="Max new tokens", info="Max new tokens", interactive=True)
                with gr.Row():
                    slider_top_k = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="Top K tokens to sample from", info="Top K tokens", interactive=True)
            
            with gr.Row():
                llm_progress = gr.Textbox(label="LLM model initialization", value="None")
            
            with gr.Row():
                qachain_btn = gr.Button("Initialize LLM model")
        
        with gr.Tab("Step 4 - Chat with PDF documents"):
            with gr.Row():
                
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(label="Chat with PDF document")
                    txt_message = gr.Textbox(label="Enter your question", placeholder="Type message (e.g. 'What is this document about?')", container=True)
                    send_button = gr.Button(value="Send", variant="primary")
                    clear_btn = gr.ClearButton(value="Clear conversation", variant='secondary')
                    # response = gr.Textbox(label="Context from document")
                    with gr.Accordion("Advanced - Document references", open=False):
                        with gr.Row():
                            doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                            source1_page = gr.Number(label="Page", scale=1)
                
                with gr.Column(scale=1):
                    image_display = gr.Image(label="Context page image")
            
        
        #define actions
        db_btn.click(
            initialize_database, 
            inputs=[document, slider_chunk_size, slider_chunk_overlap],
            outputs=[vector_db, collection_name, db_progress],
        )

        qachain_btn.click(
            initialize_LLM, 
            inputs=[llm_btn, slider_llm_temperature, slider_max_tokens, slider_top_k, vector_db, system_prompt_input],
            outputs=[qa_chain, llm_progress],
        )

        send_button.click(
            conversation, 
            inputs=[qa_chain, txt_message, chatbot],
            outputs=[qa_chain, txt_message, chatbot, doc_source1, source1_page, image_display],
        )
    
        clear_btn.click(
            clear_conversation,
            inputs=[qa_chain],
            outputs=[qa_chain, txt_message, chatbot, doc_source1, source1_page, image_display]
        )

        # with gr.Tab("Step 4 - Chatbot"):
        #     chatbot = gr.Chatbot(height=300)
            
        #     with gr.Accordion("Advanced - Document references", open=False):
        #         with gr.Row():
        #             doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
        #             source1_page = gr.Number(label="Page", scale=1)
                    
        #         with gr.Row():
        #             doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
        #             source2_page = gr.Number(label="Page", scale=1)
                    
        #         with gr.Row():
        #             doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
        #             source3_page = gr.Number(label="Page", scale=1)
            
        #     with gr.Row():
        #         msg = gr.Textbox(placeholder="Type message (e.g. 'What is this document about?')", container=True)
            
        #     with gr.Row():
        #         submit_btn = gr.Button("Submit message")
        #         clear_btn = gr.ClearButton([msg, chatbot], value="Clear conversation")
            
        # # Preprocessing events
        # #upload_btn.upload(upload_file, inputs=[upload_btn], outputs=[document])
        # db_btn.click(
        #     initialize_database, 
        #     inputs = [document, slider_chunk_size, slider_chunk_overlap], 
        #     outputs = [vector_db, collection_name, db_progress]
        # )
        
        # qachain_btn.click(
        #     initialize_LLM, 
        #     inputs = [llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], 
        #     outputs = [qa_chain, llm_progress]).then(lambda:[None, "", 0, "", 0, "", 0], 
        #     inputs = None, 
        #     outputs = [chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], 
        #     queue = False
        # )

        # # Chatbot events
        # msg.submit(
        #     conversation, 
        #     inputs = [qa_chain, msg, chatbot], 
        #     outputs = [qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], 
        #     queue = False,
        # ) # .success(render_file, inputs=[document, doc_source1], outputs=[show_img])
        
        # submit_btn.click(
        #     conversation, 
        #     inputs = [qa_chain, msg, chatbot], 
        #     outputs = [qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], 
        #     queue = False
        # ) # .success(render_file, inputs=[document, doc_source1], outputs=[show_img])
        
        # clear_btn.click(
        #     lambda:[None,"",0,"",0,"",0], 
        #     inputs = None, 
        #     outputs = [chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], 
        #     queue = False
        # )
        
    demo.queue().launch(debug=True, allowed_paths=[absolute_path])


if __name__ == "__main__":
    demo()
