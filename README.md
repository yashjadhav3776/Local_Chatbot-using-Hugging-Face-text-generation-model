# Local CLI Chatbot


## Overview
A minimal command-line chatbot that runs a small Hugging Face causal language model locally and maintains short-term conversational memory using a sliding window.


## Files
- `model_loader.py` — loads model & tokenizer
- `chat_memory.py` — sliding-window memory
- `interface.py` — CLI loop
- `requirements.txt`


## Setup
1. (Recommended) Create a virtual environment:


```bash
python -m venv venv
source venv/bin/activate # Linux / macOS
venv\Scripts\activate # Windows
