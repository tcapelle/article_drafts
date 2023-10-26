# How to fine-tune an LLM Part 3: Using Huggingface the Trainer

In this article we will re-do what we did in the previous article, but this time using the Huggingface Trainer. The Trainer is a high-level API that takes care of most of the training loop for us. It is very easy to use and it is very flexible. It is also very fast, as it uses PyTorch under the hood.

We will specifically use the SFTTrainer from the `trl` library. This is a library speciallized on reinforcement learning but has many useful features for instruction finetuning.

## Preparing the dataset

The trl library provides a set of tools to do the preprocessing of our dataset. It has built in methods to pack and tokenize the dataset in a lazy way, this is very handy when the dataset are big and you cannot load everything at once. The library provides a ConstantLengthDataset class (https://github.com/huggingface/trl/blob/02f5c1d8cee73045c837d01d7f1577a57779b035/trl/trainer/utils.py#L482) that dynamically iterates your dataset, applies the format_function and pack the sequences at your given length. This is very useful when you have a dataset with variable length sequences and you want to train with batches of fixed length. 

What is even cooler, is that you don't need to explicitely create the dataset, as the trainer will do it for you. You just need to pass the path to your dataset and the trainer will take care of the rest. More on this later.

So now, to load our dataset we will need to provide:

- The dataset with the instructions and responses. We will retrieve the same split as the one we used before.
```python
import wandb
wandb.init(project="alpaca_ft", # the project I am working on
           tags=["hf_sft"]) # the Hyperparameters I want to keep track of
artifact = wandb.use_artifact('capecape/alpaca_ft/alpaca_gpt4_splitted:latest', type='dataset')
artifact_dir = artifact.download()

from datasets import load_dataset
alpaca_ds = load_dataset("json", data_dir=artifact_dir)

```
- The formatting function to create the prompt:

```python
def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)

def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)

def create_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)

```

- Note: The tokenizer doesn't need to be provided as the `Trainer` will grab one accordingly to the model used.

## Preparing the model

To create a baseline, we will instantiate the same model as before:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'meta-llama/Llama-2-7b-hf'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)
```

we will freeze most of the layers and embeddings:

```python
n_freeze = 24

# freeze layers (disable gradients)
for param in model.parameters(): param.requires_grad = False
for param in model.lm_head.parameters(): param.requires_grad = True
for param in model.model.layers[n_freeze:].parameters(): param.requires_grad = True

# Just freeze embeddings for small memory decrease
model.model.embed_tokens.weight.requires_grad_(False);
```

## The SFTTrainer from trl

We will use the SFTTrainer class, this is a Supervised Fine Tuning Trainer specifically designed for fine-tunining LLMs. It has all the preprocessing built-in and it is a light wrapper around the original transformers' Trainer class.

```
from transformers import TrainingArguments
from trl import SFTTrainer

batch_size = 32  # this is for an A100
total_num_steps = 11_210 // batch_size  # currently there is a bug about the total number of steps reported by the progress bar

output_dir = "/tmp/transformers"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//4,
    bf16=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=total_num_steps // 10,
    max_steps=total_num_steps,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    eval_steps=total_num_steps // 3,
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="no",
)   
```

We are defining the same hyperparameters as before, as the training is fairly short, we don't save the model automatically. We will save it manually at the end of the training.

Now we can instantiate the SFTTrainer class:
```python
trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    packing=True,
    max_seq_length=1024,
    args=training_args,
    formatting_func=create_prompt,
)
```
So let's explain what is happening here with each parameter:
- `model`: the model we want to train, nothing new here.
- `train_dataset`, `eval_dataaset`: the dataset we want to use for training. This is the raw Alpaca dataset with the corresponding splits. The trainer will take care of the preprocessing.
- `packing`: this is a boolean that tells the trainer to use the ConstantLengthDataset class to pack the dataset. This is very useful when you have a dataset with variable length sequences and you want to train with batches of fixed length. This is what we had to implement manually in the previous article.
- `max_seq_length`: this is the maximum length of the sequences we want to train on. 
- `args`: the training arguments we defined before.
- `formatting_func`: this is the function that will be applied to each row of the dataset to create the prompt. This is the same function we used before. This is passed to the underlying ConstantLengthDataset class before packing.

## Sampling from the model during training

One extra step we will implement, is sampling generation of the model during training. To do so, we will use a `callback` function. This is a function that will be called at each step of the training. We will use it to generate samples from the model and log them to wandb. This is very useful to see how the model is doing during training.

We will inherit from the `WandbCallback` function that is already present in the transformers' integration and modify the evaluation step so we log the model predictions on a Table.

```python
def _generate(prompt, model, tokenizer, gen_config):
    "Call the model and generate and decode the output"
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        output = model.generate(tokenized_prompt, 
                                generation_config=gen_config)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256):
        super().__init__()
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
        tokenizer = AutoTokenizer.from_pretrained(trainer.model.name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        self.generate = partial(_generate, 
                                model=trainer.model, 
                                tokenizer=tokenizer, 
                                gen_config=self.gen_config)

    def log_generations_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt[-1000:])
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        self._wandb.log({"sample_predictions":records_table})
    
    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        self.log_generations_table(self.sample_dataset)
```

We will instantiate the callback and pass the desired dataset from which we want to sample. We will also pass the number of samples we want to generate and the maximum number of tokens we want to generate. This is useful to avoid generating very long sequences. As I don't have another split, we will sample from the `eval_dataset`.

We need to remove the answers from the dataset, as we don't want the model to cheat. We will do this by creating a new dataset with the same format as the original one, but with empty answers. We can re-use the `create_prompt` function we defined before.

```python
# remove answers
def create_prompt_no_answer(row):
    row["output"] = ""
    return {"text": create_prompt(row)}

test_dataset = eval_dataset.map(create_prompt_no_answer)

```

we then instantiate the callback and add it to the Trainer:

```python

wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=10, max_new_tokens=256)
trainer.add_callback(wandb_callback)
```

We are now ready to train the model:

```python
trainer.train()
wandb.finish()
```
