import glob
from logger import Logger
from mlx_lm import load, generate
from dataset import csv_creator

### Initialize model for inference
model_name1 = 'unsloth' #Llama
model_name2 = 'fused_model_4bit' #Fused model

### Load file
location = input('Enter file location: ')# Location format should be ./location/file.csv
files = csv_creator.read_table(location)
prompt = f'Analyze the company from investor prespective. Financial data {files}'

### Load model and generate response
model, tokenizer = load(model_name1)
response = generate(model, tokenizer, prompt= prompt, max_tokens=1024, temp=0.5, verbose=True)