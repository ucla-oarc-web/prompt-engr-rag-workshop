# Ollama Setup and Installation

Windows  
Ollama download page: https://ollama.com/download  

MAC (M series chips)  
brew install ollama   
ollama serve - start ollama  
ollama pull llama3  
ollama pull nomic-embed-text  
ollama pull llama2 (optional)  
ollama list - list available models  
pip install ollama - Install Ollama Python package  

Linux  
curl -fsSL https://ollama.com/install.sh | sh  
ollama serve - start ollama  
ollama pull llama3  
ollama pull nomic-embed-text  
ollama pull llama2 (optional)  
ollama list - list available models  
pip install ollama - Install Ollama Python package  

# Workshop UI App Setup and Installation

Create the app folder  
mkdir workshop-rag  
cd workshop-rag  

Create virtual environment   
python -m venv venv   

Activate virtual environment   
On macOS/Linux:   
source venv/bin/activate  

Install needed pip packages  
pip install --upgrade pip   
pip install langchain langchain-community langchain-core langchain-huggingface streamlit PyPDF2 faiss-cpu sentence-transformers transformers torch python-dotenv beautifulsoup4 lxml  

Then install the rest using requirements.txt  
pip install -r requirements.txt  

# Run Workshop UI App 
streamlit run streamlit_app_workshop.py

Once the app is running  
1.) Select Model and Embeddings. You can keep the default selection  
2.) Update the System Prompt as needed  
3.) Click on "Add Data Source" => Enter Webpage URL and click Add URL or Upload PDF and click on Run PDF  
4.) Once popup window closes, the app is ready and you can start chatting. Enter the prompt message and hit enter 

