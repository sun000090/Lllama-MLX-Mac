from dataset import csv_creator
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.tuner import train, evaluate, TrainingArgs
from mlx_lm.tuner import linear_to_lora_layers
import tqdm
import json
from pathlib import Path
from logger import Logger

logs = Logger.logger_init()

class TrainModel:
    # Prepare the prompt template
    def generate_prompt(data_point):
        try:
            data_point['prompt'] = data_point['prompt'].replace('\\','')
            data_point['response'] = data_point['response'].replace('\\','')
            result = f"{data_point['prompt']}\n{data_point['response']}"
            logs.info('Data generated')
            return result
        except Exception as e:
            logs.info(f'Data generation failed {e}')
            return None
        
    # Function to add a 'text' column based on the generated prompt
    def add_text(example):
        try:
            example["text"] = TrainModel.generate_prompt(example)
            # Optionally remove the original columns to clean up the dataset
            example.pop("prompt", None)
            example.pop("response", None)
            logs.info('Data generated')
            return example
        except Exception as e:
            logs.info(f'Data generation failed {e}')
            return None
    
    # Process dataset for model training
    def preprocess(dataset):
        return [t["prompt"] + "\n" + t["response"] for t in dataset]

    def preprocess_fin(data_set):
        try:
            data_set_ = map(TrainModel.preprocess, data_set)
            logs.info('Dataset preprocessed')
            return data_set_
        except Exception as e:
            logs.info(f'Dataset preprocessing error {e}')
            return None

    # Training pipeline
    def train_model(model_name_,dataset1,dataset2,seq_length):
        try:
            # Make a directory to save the adapter config and weights
            adapter_path = Path("adapters")
            adapter_path.mkdir(parents=True, exist_ok=True)
            logs.info('Adapter path created')

            # Lora config
            lora_config = {
            "num_layers": 7,
            "lora_parameters": {
                "rank": 8,
                "scale": 16.0,
                "dropout": 0.0,
                "keys": ["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj",
                        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
            }}

            # Save the LoRA config to the adapter path
            with open(adapter_path / "adapter_config.json", "w") as fid:
                json.dump(lora_config, fid, indent=4) 
            logs.info('Adapter file written')   

            # Training arguments
            training_args = TrainingArgs(
                adapter_file=adapter_path / "adapters.safetensors",
                max_seq_length=seq_length,
                batch_size=2,
                iters=int(len(dataset1)/2),
                steps_per_eval=50,
                steps_per_save=50)
            logs.info('Training parameter initialized')
            
            # Load model
            model, tokenizer = load(model_name_)
            logs.info('Model loaded')

            # Freeze the base model
            model.freeze()

            # Convert linear layers to lora layers
            linear_to_lora_layers(model, lora_config["lora_layers"], lora_config["lora_parameters"])
            logs.info('Lora loaded')

            # Number of training parameters
            num_train_params = (sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
            logs.info(f'Number of trainable parameters {num_train_params}')

            # Put the model in training mode:
            model.train()

            # Make the optimizer:
            opt = optim.AdamW(learning_rate=1e-3)

            # Make a class to record the training stats:
            class Metrics:
                train_losses = []
                val_losses = []
                def on_train_loss_report(self, info):
                    self.train_losses.append((info["iteration"], info["train_loss"]))
                def on_val_loss_report(self, info):
                    self.val_losses.append((info["iteration"], info["val_loss"]))

            metrics = Metrics()

            logs.info('Training started')
            # Train model:
            train(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                optimizer=opt,
                train_dataset=dataset1,
                val_dataset=dataset2,
                training_callback=metrics,
            )
            logs.info('Training completed')
            return
        
        except Exception as e:
            logs.info(f'Model training failed {e}')
            return None
