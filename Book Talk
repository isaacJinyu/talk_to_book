from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
# Get your API keys from openai, you will need to create an account. 
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
os.environ["OPENAI_API_KEY"] = "输入你的key"
# location of the pdf file/files. 
reader = PdfReader('D:\Prompt Engineer\Pure Mathematics for Beginners A Rigorous Introduction to Logic, Set Theory, Abstract Algebra, Number Theory, Real Analysis,... (Steve Warner) (Z-Library).pdf')
# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
        # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
openai = OpenAI(model="text-davinci-003")
chain = load_qa_chain(openai, chain_type="stuff")
# use a while loop to keep asking questions until quit
while True:
  # get the question from user input
  query = input("请输入你的问题，或者输入quit退出：")
  # check if the input is quit
  if query == "quit":
    # break the loop and exit the program
    break
  else:
    # search for relevant documents based on the question
    docs = docsearch.similarity_search(query)
    # run the question answering chain and print the answer
    answer = chain.run(input_documents=docs, question=query, max_tokens=1000)
    print(answer)
