from transformers import pipeline
from datasets import load_dataset
from huggingface_hub import login
import torch
import json
import os
import datetime

# Add code to Login to Hugging Face


modelid = "meta-llama/Llama-3.3-70B-Instruct"
login(token=token)

print("Login done", flush=True)

# Dataset list
dataset_list = [
  "Rulebert-Union-Rules",
  "abduction_animal",
  "abduction_person",
  "alpha_nli",
  "adv",
  "anli",
  "babi_task_15",
  "babi_task_16",
  "bigbench-logical-Args",
  "bigbench_deduction",
  "birdelectricity",
  "cluttr",
  "cluttr_systematic",
  "folio",
  "logiQA",
  "logiQA_2.0",
  "logicNLI",
  "natlang",
  "prontoqa",
  "proofwriter",
  "reclor",
  "rulebert",
  "wanli",
  "winologic"
]

batch_size = 32



# Initialize the pipeline
generator = pipeline(
    "text-generation",
    model=modelid,
    tokenizer=modelid,
    device_map="auto",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16},
    token=token
)
generator.tokenizer.pad_token = generator.tokenizer.eos_token

for dataset_name in dataset_list:
    # Load dataset
    print(dataset_name)
    print("\n")
    print(datetime.datetime.now())
    print("\n")
    ds = load_dataset("logicreasoning/logi_glue", dataset_name)
    ds_type = "train"
    if "train" not in ds.keys():
        ds_type = "test"

    # Define output file path
    file_path = f"Rational_Generations/Llama3.3_Outputs/rationale_data_llama3.3_{dataset_name}.json"

    # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    print("\n Length of Dataset", len(ds[ds_type]))
    # Process dataset in batches of batch_size
    for i in range(0, len(ds[ds_type]), batch_size):
        # Get the current batch
        print("\n Current bacth(dataset number )", i)
        train_data = ds[ds_type][i:i + batch_size]

        # Create prompt texts for the current batch
        prompt_texts = [
            f"INPUT: {input_text}. ANSWER: {answer_text}. GENERATE proper rationale to reach this answer."
            for input_text, answer_text in zip(train_data["input"], train_data["answer_text"])
        ]

        print(
            "\n ............................................................................................................",
            flush=True)

        # Generate responses in batch using the generator
        outputs = generator(prompt_texts, max_length=100000, pad_token_id=generator.tokenizer.pad_token_id,
                            truncation=True)

        # Process each prompt and corresponding output
        for index, output in enumerate(outputs):
            generated_text = output[0]["generated_text"]
            rationale = generated_text[len(prompt_texts[index]):]

            print(f"Dataset: {dataset_name}\nPrompt: {prompt_texts[index]}\nRationale: {rationale}", flush=True)
            print("\n-------------------------------------------\n", flush=True)

            data.append({
                "Category": dataset_name,
                "Input": train_data["input"][index],
                "Rationale": rationale,
                "Output": train_data["answer_text"][index]
            })

            # Save results incrementally to avoid data loss
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

print("Processing complete.", flush=True)

