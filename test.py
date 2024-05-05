# Store llm api keys in the .env file.
from dotenv import load_dotenv

#Utility packages based on the usecase
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder

# Get access to the llm . In this case, we are using the llm from openai. API key should be defined in the .env
from langchain_openai import ChatOpenAI

#Access the vector store. FAISS is an in-memory vector store.
from langchain_community.vectorstores.faiss import FAISS

# To generate embeddings for the vector store. These will be used as context within the prompt template
from langchain_openai import OpenAIEmbeddings

# import prompts to generate a prompt template, that will take context(document embeddings) and the user question.
from langchain_core.prompts import ChatPromptTemplate

#For conversational retrieval . First create a conversational retriever
from langchain.chains import create_history_aware_retriever

#For final retrieval chain . Pass history aware retriever + document chain.
from langchain.chains import create_retrieval_chain

load_dotenv()
llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()
output_parser = StrOutputParser()

#Creating a simple chain. Without our own context, making use of data the llm already has.
# prompt = ChatPromptTemplate.from_messages([
#     ("system","You are an English-French translator that return whatever the user says in french"),
#     ("user","{input}")
# ])
# chain = prompt | llm

# Make use of output_parser
# chain = prompt | llm | output_parser

# res = chain.invoke({"input":"I enjoy going to gym"})


# Retrieval chain
loader = WebBaseLoader("https://blog.langchain.dev/langchain-v0-1-0/")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(documents=docs)
vectorstore = FAISS.from_documents(documents=docs,embedding=embeddings)

# template = """Answer the following question based only on the provided context:
# <context>
# {context}
# </context>
# Question: {input}
# """
# prompt = ChatPromptTemplate.from_template(template)

# #create chain for documents
# document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)

#Test if the retrival chain works with a test context from a fake document snippet.
# from langchain_core.documents import Document
# test_context = [Document(page_content="0.1.0 is the new version of a llm app development framework")]

# res = document_chain.invoke({
#     "input":"What is langchain 0.1.0",
#     "context":test_context
# })


#Create the final retrieval chain, chaining the context, prompt and the llm.
#this not a conversational retrieval chain, as the model doesnt remember our conservation so, 
#it cannot answer based on the older context.

#Get the first element, which is the document embeddings.
retriever = vectorstore.as_retriever()

#Retrieval chain 
# retrieval_chain = create_retrieval_chain(retriever,document_chain)

#This will first get releval document from the vector store and then pass it as a context along with the user question to the llm
# response = retrieval_chain.invoke({
#     "input":"What is new in langchain 0.1.0"
# })
# print(response['answer'])




# Conversional retrieval chain.
# This get documents not only relevant to the question asked, but the conversational history
# This will also take into account the conversation history itself to build the context.

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

conversational_retrieval_chain = create_history_aware_retriever(llm, retriever,prompt)


# chat_history = [
#     HumanMessage(content="Is there anything new about langchain 0.1.0?"),
#     AIMessage(content="Yes!")
# ]

#Get documents relevant to the entire conversation.
# res = retriever_chain.invoke({
#     "chat_history":chat_history,
#     "input":"Tell me more about it!"
# })

#Create a prompt for the final retrieval chain to feed above chain to the llm for final answer
#This is a shorter version of the previous prompt we created from_template
# template = """Answer the following question based only on the provided context:
# <context>
# {context}
# </context>
# Question: {input}
# """
# prompt = ChatPromptTemplate.from_template(template)


prompt  = ChatPromptTemplate.from_messages([
    ("system","Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    "user","{input}"
])

document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)

#Below call takes retriever which can be the vector store or the retriever chain previously formed with conversational chain
final_retrieval_chain=create_retrieval_chain(conversational_retrieval_chain,document_chain)
res = final_retrieval_chain.invoke({
    "chat_history":[],
    "input":"What is langchain 0.1.0 about?"
})


#Create a mock chat history to test if the model is conversational
#Create a mock history conversation to test conversational chain
from langchain_core.messages import HumanMessage, AIMessage
chat_history = [
    HumanMessage(content="Is there anything new about langchain 0.1.0?"),
    AIMessage(content="Yes!")
]

res = final_retrieval_chain.invoke({
    "chat_history":chat_history,
    "input":"Tell me more about it"
})

print(res['answer'])