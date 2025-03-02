from vllm import LLM, SamplingParams
from datasets import load_dataset
from huggingface_hub import login
import json
import os
import datetime


model_id = "meta-llama/Llama-3.3-70B-Instruct"


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

batch_size = 4  # Increase batch size to utilize multiple GPUs

# Initialize the VLLM model with multi-GPU support
llm = LLM(
    model=model_id,
    tensor_parallel_size=4
    #dtype="bfloat16",
    #trust_remote_code=True,
    #device_map="auto"  # Automatically distribute across all available GPUs
)
print("Test Strated")

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
    file_path = f"/home/cr8dl-user/palash/Rational_Generations/Llama3.3_Outputs/rationale_data_llama3.3_{dataset_name}.json"

    # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

        # Process dataset in larger batches
    for i in range(0, len(ds[ds_type]), batch_size):
        train_data = ds[ds_type][i:i + batch_size]

        # Create prompt texts
        prompt_texts = [
            f"INPUT: {input_text}. ANSWER: {answer_text}. GENERATE proper rationale to reach this answer."
            for input_text, answer_text in zip(train_data["input"], train_data["answer_text"])
        ]

        # Print processing info
        print(
            "\n ............................................................................................................",
            flush=True)

        # Define sampling parameters for VLLM
        sampling_params = SamplingParams(
            max_tokens=10000,
            # Keeping max tokens moderate for large batches
            temperature=0.0,
            top_p=1.0,
            stop=["ANSWER:"]
        )

        # Generate responses using VLLM
        outputs = llm.generate(prompt_texts, sampling_params)


        print(outputs)

        # Process each prompt and corresponding output
        for index, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            rationale = generated_text
            #rationale = generated_text[len(prompt_texts[index]):]
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


print("Processing complete. ", flush=True)

