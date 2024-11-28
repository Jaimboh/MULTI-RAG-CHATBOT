# MULTI-RAG-CHATBOT
This project is an AI-powered chatbot capable of interacting with various data, including images, grocery datasets, PDFs, and standard queries. The assistant utilizes vector databases, advanced embedding models, and  LLM 
 Routing for effective responses depending on the user input.

---

## Key Features

- **Image Dataset Search**: Perform semantic searches over an image dataset stored in a vector database.
- **Grocery Data Interaction**: Query a grocery dataset and retrieve details like prices, categories, and nutritional values.
- **Fine-tuning Information**: Answer questions using PDF data embedded in a vector database.
- **General Assistance**: Handle everyday queries and offer helpful responses.
- **LLM - cost tracking and evaluation using langtrace**:Langtrace is an open-source observability tool that collects and analyze traces in order to help you improve your LLM apps[resource](https://docs.langtrace.ai/introduction)

---



## Installation Guide

### 1. Clone the Repository

Clone the project to your local machine:
```bash
git clone https://github.com/Jaimboh/MULTI-RAG-CHATBOT.git
cd MULTI-RAG-CHATBOT
```
### 2. Create a virtual environment (optional but recommended):
``` bash
python -m venv venv

source venv/bin/activate
```
#### On windows to activate the virtual environment use:
``` bash
venv\Scripts\activate
```
### 3.If you are using a conda environment this is how to go about it:
```bash
conda create -n multirag -y
conda activate multirag
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```
### 4. Run the application
```bash
streamlit run main.py
```

---
### Resources
1.https://chat.notdiamond.ai/

2.https://github.com/Not-Diamond/notdiamond-examples

3.https://openrouter.ai/

4.https://github.com/Scale3-Labs/langtrace