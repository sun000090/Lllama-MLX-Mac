from dataset import csv_creator
import glob
from logger import Logger
from dataset import Dataset
from train_cpu_struct import TrainModel

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
!ls adapters/

### Fuse the adaptors
!mlx_lm.fuse --model {model_name1}

### Quantize and save
!mlx_lm.convert --hf-path {model_name1} --mlx-path {model_name1} -q
