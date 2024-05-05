Packages installed
langchain
langchain_openai
beautisoup4
python3-dotenv

Process---
1. Get llm
2. Get documents
3. Split documents
4. Create vector store for embeddings
5. Create retriever => vectorstore.as_retriever()

####### for non conversational create_retrieval_chain
6. Create 1st chain from llm + prompt(input,context)
7. Create second chain from vectorstrore retriever + first chain

######### for conversational
8. Generate 1st prompt (chat_history+input) (no document context passed)
9. Create history aware retriever chain (create_history_aware_retriever) from llm+retriever+prompt(chat_history+input)

10. Generate 2nd prompt (context+chat_history+input)
11. Create document chain(llm+prompt) (document context passed)

12. Create final chain (create_retrieval_chain)passing the runnables of the history aware retriever chain + document chain
10. Generate 2nd prompt()
