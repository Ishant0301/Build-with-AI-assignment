import os
import tempfile
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader , PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class RAGChatbot:
    """
    A Retrieval-Augmented Generation (RAG) based chatbot that answers questions
    using information from user-uploaded files with Google's Gemini model.
    """
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.initialized = False
    
    def initialize_with_file(self, file_path, file_type):
        """Initialize the chatbot with a user-uploaded file."""
        try:
            self._load_data(file_path, file_type)
            self._setup_embeddings()
            self._create_vectorstore()
            self._setup_qa_chain()
            self.initialized = True
            return True
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            return False
    
    def _load_data(self, file_path, file_type):
        """Load and preprocess the uploaded file."""
        try:
            # Select appropriate loader based on file type
            if file_type == "csv":
                loader = CSVLoader(file_path=file_path)
            elif file_type == "pdf":
                loader = PyPDFLoader(file_path=file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path=file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.texts = text_splitter.split_documents(self.documents)
            
            st.success("File loaded and processed successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            raise
    
    def _setup_embeddings(self):
        """Initialize the embeddings model."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            st.error(f"Error setting up embeddings: {str(e)}")
            raise
    
    def _create_vectorstore(self):
        """Create a vector store from the document embeddings."""
        try:
            self.vectorstore = FAISS.from_documents(
                self.texts,
                self.embeddings
            )
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            raise
    
    def _setup_qa_chain(self):
        """Set up the retrieval QA chain with Gemini."""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3
            )
            
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            {context}
            
            Question: {question}
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        except Exception as e:
            st.error(f"Error setting up QA chain: {str(e)}")
            raise
    
    def query(self, question):
        """Query the chatbot with a question."""
        try:
            if not self.initialized:
                return {"result": "Chatbot not initialized. Please upload a file first.", "source_documents": []}
            
            result = self.qa_chain({"query": question})
            return result
        except Exception as e:
            st.error(f"Error querying the chatbot: {str(e)}")
            return {"result": "Sorry, I encountered an error processing your question.", "source_documents": []}

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def main():
    """Streamlit application for the RAG chatbot."""
    st.title("📄 Build with AI Chatbot")
    st.markdown("Upload your file and ask questions about its content")
    
    # Initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your file (CSV, PDF, or TXT)",
        type=["csv", "pdf", "txt"]
    )
    
    if uploaded_file is not None:
        # Display file info
        file_type = uploaded_file.name.split(".")[-1].lower()
        st.info(f"Uploaded {file_type.upper()} file: {uploaded_file.name}")
        
        # Save file and initialize chatbot
        with st.spinner("Processing your file..."):
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                success = st.session_state.chatbot.initialize_with_file(file_path, file_type)
                if success:
                    st.success("Chatbot is ready! Ask questions about your file.")
                # Clean up temporary file
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input (only show if chatbot is initialized)
    if st.session_state.chatbot.initialized:
        if prompt := st.chat_input("Ask me anything about your file"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get chatbot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.query(prompt)
                    answer = response["result"]
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show sources if available
                    if response["source_documents"]:
                        with st.expander("See sources"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Source {i+1}:** {doc.page_content[:200]}...")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
    elif uploaded_file is None:
        st.info("Please upload a file to begin chatting")
    else:
        st.warning("Processing your file... please wait")

if __name__ == "__main__":
    main()