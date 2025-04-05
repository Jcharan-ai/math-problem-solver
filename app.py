import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_API_KEY"]=os.getenv("HF_API_KEY")

def create_documents(docs):
    """Create documents from uploaded files."""
    
    loaders = {
        "txt": TextLoader,
        "pdf": PyMuPDFLoader,
        "docx": UnstructuredWordDocumentLoader,
    }

    documents = []
    for doc in docs:
        os.makedirs("./temp", exist_ok=True)
        tempdocs = os.path.join("./temp", doc.name)
        with open(tempdocs, "wb") as f: 
            f.write(doc.getbuffer())
        file_extension = f.name.split(".")[-1]
        if file_extension in loaders:
            loader = loaders[file_extension](f"./temp/" + doc.name)
            loaded_docs = loader.load()
            for loaded_doc in loaded_docs:
                # Extract page_content and metadata properly
                page_content = loaded_doc.page_content if hasattr(loaded_doc, "page_content") else ""
                metadata = loaded_doc.metadata if hasattr(loaded_doc, "metadata") else {}
                documents.append(Document(
                    page_content=page_content,
                    metadata=metadata
                ))
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    return docs

def create_embeddings(docs):
    """Create embeddings for the documents."""
    # Ensure docs have valid page_content
    for doc in docs:
        if not isinstance(doc.page_content, str):
            raise ValueError(f"Invalid page_content: {doc.page_content}")

    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore

# Set up the Streamlit app
st.set_page_config(page_title="Math problem solver", page_icon=":robot:")
st.title("Math problem solver with Groq")
st.write(
    "This app uses Groq to solve math problems and answer questions. You can ask it to solve math problems or provide information on a topic."
)

st.sidebar.title("settings")   
groq_api_key=st.sidebar.text_input("Enter your Groq API key here:", type="password", key="groq_api_key") 

if not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.stop()
    
# Initialize the Groq LLM
llm = ChatGroq(model_name="gemma2-9b-it", api_key=groq_api_key)

wiki_wrapper=WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="Search Wikipedia for information on a topic.",
)
llm_math = LLMMathChain.from_llm(llm, verbose=True)
calculator_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="Use a calculator to solve math problems.",
)

prompt="""
You are a helpful assistant that can solve math problems and answer questions. You can use a calculator to solve math problems and search Wikipedia for information on a topic.
question: {input}
"""

prompt_template = PromptTemplate(
    input_variables=["input"],
    template=prompt,
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
)

llm_chain_tool = Tool(
    name="LLMChain",
    func=llm_chain.run,
    description="Use a language model to answer questions.",
)

uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
#create docs
docs=create_documents(uploaded_file)
    
# Create embeddings
vectorstore = create_embeddings(docs)

retriver=vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriver)

# Use the RetrievalQA Chain as a Tool
retriver_tool = Tool(
    name="My Vector Store",
    func=qa_chain.run,
    description="Useful for answering questions based on the vector store."
)

tools = [wiki_tool, calculator_tool, llm_chain_tool,retriver_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
        
chat_input = st.chat_input("Ask a question or solve a math problem:")
if chat_input:
    st.session_state["messages"].append({"role": "user", "content": chat_input})
    st.chat_message("user").write(chat_input)
    
    with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container())
            response = agent.run(input=chat_input,callbacks=[st_cb])
            st.session_state["messages"].append({"role": "assistant", "content": response})
                        
            st.write(response)