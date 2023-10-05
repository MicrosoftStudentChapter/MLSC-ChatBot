import pandas as pd
import google.generativeai as palm
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
# from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
from langchain.llms import VLLM
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
import gradio as gr
import requests
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
os.environ['GOOGLE_API_KEY']=" "

# models = [m for m in palm.list_models() if "generateText" in m.supported_generation_methods]
# model = models[0].name
print('Imports Done')


db_path = 'C:/Users/HP/PycharmProjects/MLSCBot/venv/vectordb/db_faiss'

# print('Reading Document')
# os.mkdir('/home/Sparsh/data')
# url = 'https://ia803106.us.archive.org/13/items/Encyclopedia_Of_Agriculture_And_Food_Systems/Encyclopedia%20of%20Agriculture%20and%20Food%20Systems.pdf'
# response = requests.get(url)
# with open('/home/Sparsh/data/document.pdf', 'wb') as f:
#          f.write(response.content)

print('Creating Chunks')


loader = DirectoryLoader('C:/Users/HP/PycharmProjects/MLSCBot/venv/MLSCBot',glob = "*.pdf",loader_cls = PyPDFLoader)
data = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 100)
chunks = splitter.split_documents(data)

print('Mapping Embeddings')
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(model_name=model_name,
    encode_kwargs=encode_kwargs)
embeddings = model_norm
db = FAISS.from_documents(chunks,embeddings)
db.save_local(db_path)
# db = FAISS.load_local(db_path,embeddings)
print('Prompt Chain')

custom_prompt_template = """You are a helpful bot designd for MLSC TIET that is Microsoft Student Learn Chapter,TIET which a technical society for thir website your task is to answer all queries about MLSC every answer you provide should be i context of MLSC if any question is not in that context then yyou should ecline that question by saying 'It is out of context',if you don't know the answer don't try to make it up just politely decline that question,you can extrapolayte the things a little just to be more informative but dont sound boasty and exaggerating say something else out of the context of the document,don't answer any questions that pertain to any specific persons and if questions about roles demnad names of position holders of MLSC give a general description of role instead of person
You can accept some basic greetings to interact with the user but be sure to remisn in context of MLSC only
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



print('Creating LLM')

llm2 = GooglePalm(
           max_new_tokens=1024,
           top_k=10,
           top_p=0.5,
           temperature=0.5)

print(llm2("What is the capital of France ?"))
# qa_chain = ConversationalRetrievalChain.from_llm(llm2,retriever=db.as_retriever(search_kwargs={'k': 2}),
#                                        return_source_documents=False,
#                                        memory=memory)
qa_chain = RetrievalQA.from_chain_type(llm=llm2,
                                      chain_type='stuff',
                                      retriever=db.as_retriever(search_kwargs={'k': 5}),
                                      return_source_documents=False,
                                      chain_type_kwargs={'prompt': prompt})
history_df = pd.DataFrame(columns = ['Question','Answer'])
def qa_bot(query):
  global history_df
  response = qa_chain({'query': query})
  print(response)
  response_df = pd.DataFrame.from_dict([response])
  print(response)
  response_df.rename(columns = {'query' : 'Question','result' : 'Answer'},inplace = True)
  print(response)
  history_df = pd.concat([history_df,response_df])
  history_df.reset_index(drop = True,inplace = True)
  print(history_df)
  return (response['result'])

with gr.Blocks(theme='upsatwal/mlsc_tiet') as demo:
    title = gr.HTML("<h1>MLSCBot</h1>")
    with gr.Row():
      img = gr.Image('C:/Users/HP/Downloads/banner.png',label = 'MLSC Logo',show_label = False,elem_id = 'image',height = 200)
    input = gr.Textbox(label="How can I assist you?")  # Textbox for user input
    output = gr.Textbox(label="Here you go:")  # Textbox for chatbot response
    btn = gr.Button(value="Answer",elem_classes="button-chatbot",variant = "primary")  # Button to trigger the agent call
    btn.click(fn=qa_bot, inputs=input,outputs=output)
demo.launch(share=True, debug=True,show_api = False,show_error = False)
