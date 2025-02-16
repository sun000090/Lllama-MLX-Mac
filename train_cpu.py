from dataset import csv_creator
import glob
from logger import Logger
from dataset import Dataset
from train_cpu_struct import TrainModel
import subprocess

logs = Logger.logger_init()

### Read all data from location
loc1 = 'Training_Context/*.csv'
loc2 = 'Training_Answer/*.txt'
training_data = csv_creator.data_creator(loc1=loc1,loc2=loc2)

### Preprocess dataset
train_set = TrainModel.preprocess(training_data['train'])
test_set = TrainModel.preprocess(training_data['test'])

### Initialize and train model
model_name1 = 'unsloth/Llama-3.2-3B-Instruct' #Llama
model_name2 = 'unsloth/DeepSeek-R1-Distill-Llama-8B' #Deepseek
trainingSave = TrainModel.train_model(model_name_=model_name1, dataset1 = train_set,
                                           dataset2 = test_set, seq_length=1024)

### Check for adapters saved or not 
subprocess.run('ls adapters/',shell=True)

fused_loc = 'fused_model'
fused_loc_4bit = 'fused_model_4bit'

### Delete any existing folder
try:
    subprocess.run(f'rm -rf {fused_loc}', shell=True)
    subprocess.run(f'rm -rf {fused_loc_4bit}', shell=True)
    subprocess.run(f'rm -rf {model_name1}', shell=True)
except Exception as e:
    pass

### Fuse the adaptors
subprocess.run(f'mlx_lm.fuse --model {model_name1}',shell=True)

### Quantize and save
subprocess.run(f"mlx_lm.convert --hf-path {fused_loc} --mlx-path {fused_loc_4bit} --quantize --q-group-size {64} --q-bits {8}",shell=True)
subprocess.run(f"mlx_lm.convert --hf-path {model_name1} --mlx-path {model_name1} --quantize --q-group-size {64} --q-bits {8}", shell=True)
