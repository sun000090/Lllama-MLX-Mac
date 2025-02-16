import glob
from logger import Logger
import subprocess

logs = Logger.logger_init()

### Load models and dequantize
model_name1 = 'unsloth/Llama-3.2-1B-Instruct' #Llama
model_name2 = 'unsloth/Llama-3.2-1B-Instruct_32bit' #Dequnatized model
model_name3 = f"{model_name2}.gguf" # gguf model

### Delete any existing folder
try:
    subprocess.run(f'rm -rf {model_name2}', shell=True)
    subprocess.run(f'rm -rf {model_name3}', shell=True)
except Exception as e:
    pass

### Save a dequantize version
subprocess.run(f"mlx_lm.convert --hf-path {model_name1} --mlx-path {model_name2} -d",shell=True)

### Install llama.cpp
subprocess.run("git clone https://github.com/ggerganov/llama.cpp.git", shell=True)
subprocess.run("pip install -r llama.cpp/requirements.txt", shell=True)

### Convert to gguf and save
subprocess.run(f"python3 llama.cpp/convert_hf_to_gguf.py {model_name2} --outfile {model_name3} --outtype q8_0", shell=True)
subprocess.run('CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python',shell=True)