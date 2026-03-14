import os
import json
import random
import argparse
from tqdm import tqdm
import sys
from datetime import datetime
import re
from huggingface_hub import login
from datasets import load_from_disk, Dataset
import pandas as pd
import time
from collections import defaultdict

HF_ACCESS_TOKEN = "xxxxx" # insert with your Hugging Face access token

# library for modelling
import transformers
import torch
from transformers import set_seed
from accelerate import Accelerator
# utils function
from utils import extract_answer, calculate_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str)
parser.add_argument("--dataset_path", default='dataset_with_shots', type=str)
parser.add_argument("--dataset_train_source", default="train_experiment1", type=str)
parser.add_argument("--dry_run", default=False, action="store_true", help="whether it's a dry run or real run.")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature of 0 implies greedy sampling.")
parser.add_argument("--num_shots", type=int, default=0, help="Jumlah maksimum contoh few-shot (default: 5)")
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--seed", type=int, default=42) #seed: 42, 943, 1454, 2921, 3264
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--chat_model", default=False, action="store_true", help="for some model only accepted chat_template format")
parser.add_argument("--max_input_tokens", type=int)


QA_TEMPLATE = """
Pertanyaan: Berdasarkan tabel di atas, {question}
Jawaban: {answer}
"""

TEMPLATE_ZEROSHOT = """
Bacalah tabel di bawah ini mengenai '{title}' untuk menjawab pertanyaan berikut.

Tabel:
{table}
Pertanyaan: Berdasarkan tabel di atas, {question}
Jawaban: adalah 
"""

TEMPLATE_FEWSHOT = """
Tabel:
{table_example}

{qa_examples}

Seperti contoh diatas, jawablah pertanyaan berikut mengenai '{title}' sesuai dengan
tabel yang diberikan.

Tabel:
{table}
Pertanyaan: {question}
Jawaban: """

system_prompt = """
Anda ahli dalam menganalisis tabel dan menjawab pertanyaan berdasarkan data tabular.
Tugas Anda adalah:
1. Membaca tabel dengan hati-hati
2. Memahami pertanyaan dan mencari jawaban yang terdapat di dalam tabel
3. Berikan jawaban dengan pola "jawaban: X" di mana X adalah jawaban akhir Anda. Perhatikan 2 hal ini juga:
    - Jika X terdiri angka dan satuan (misalnya: '2 kali lipat', '90%', '49 persen', '120 orang', '3,5 tahun'), maka cukup tuliskan hanya angkanya saja (contoh:'2', '90', '49', '120', '3.5')
    - Jika X bukan angka (misalnya: 'Sydney', 'Ya', 'Tidak' , atau lainnya), maka tuliskan teks tersebut apa adanya.
4. Jawaban akhir Anda harus singkat dan tepat.
"""

def group_dataset_by_table(dataset):
    grouped = defaultdict(lambda: {
        "id": None,
        "title": None,
        "header": None,
        "data": None,
        "source": None,
        "questions": []
    })
    for row in train_set:
        table = row["table"]
        table_id = table["id"]

        if grouped[table_id]["id"] is None:
            grouped[table_id].update({
                "id": table_id,
                "title": table.get("title"),
                "source": row.get("table_source"),
                "header": table.get("header"),
                "data": table.get("data"),
                "questions": []
            })

        grouped[table_id]["questions"].append({
            "id": row["id"],
            "question_type": row.get("question_type"),
            "question": row["question"],
            "answer_text": row.get("answer_text"),
            "answer_coordinates": row.get("answer_coordinates")
        })

    dataset_by_table = list(grouped.values())
    dataset_by_table_ds = Dataset.from_list(dataset_by_table)
    return dataset_by_table_ds

def extract_output_text(response):
    """
    Extracts the first text with type='output_text' from an OpenAI response object.
    If not found, returns 'Answer not found'.
    """
    if hasattr(response, "output"):
        for item in response.output:
            if hasattr(item, "content") and item.content:
                for content in item.content:
                    if getattr(content, "type", None) == "output_text" and hasattr(content, "text"):
                        return content.text.strip()

    return "Jawaban tidak ditemukan"

def extract_chat_model_output_text(text: str) -> str:
    """
    Extracts the answer from LLaMA-3 ChatML output.
    Grabs the text after <|start_header_id|>assistant<|end_header_id|>
    """
    # Cari bagian assistant
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Hapus newline/space di awal/akhir
        return answer
    else:
        # fallback: kembalikan seluruh teks jika tag tidak ditemukan
        return text.strip()

def build_prompt(entry, train_samples, max_shots=5):
    """Builds a few-shot or zero-shot prompt based on test & train data."""
    
    question = entry["question"]
    table = entry["table"]
    table_title = table["title"]
    table_text = pd.DataFrame(table["data"], columns=table["header"]).to_markdown(index=False)
    
    # default prompt is zero shot
    prompt = TEMPLATE_ZEROSHOT.format(table=table_text, question=question, title=table_title)
    
    if max_shots>0:
        top_ids = entry.get("top_5_similar_ids_from_train_experiment1", [])
        top_5_similar_ids_from_train_set = [ex for ex in train_samples if ex["id"] in top_ids][:5]
        table_id_ref = top_5_similar_ids_from_train_set[0]["table"]["id"] # get first sample
        
        train_set_by_table = group_dataset_by_table(train_set)
        shots = train_set_by_table.filter(lambda x: x["id"] == table_id_ref)
        
        if len(shots)>0:
            example = shots[0]
            table_example_text = pd.DataFrame(example["data"], columns=example["header"]).to_markdown(index=False)
            
            qa_example_prompt = ""
            for item in example["questions"]:
                qa_example_prompt += QA_TEMPLATE.format(
                    question=item["question"],
                    answer=", ".join(item["answer_text"]),
                )
            
            prompt = TEMPLATE_FEWSHOT.format(
                table_example=table_example_text, 
                qa_examples=qa_example_prompt,
                table=table_text,
                question=question,
                title=table_title
            )
            
    return {"prompt":prompt}

def truncate_prompts(batch_prompts, tokenizer, max_input_tokens=4096):
    """
    Truncates the prompt to ensure it does not exceed the model's input token limit.
    Returns a prompt that is safe for inference.
    """
    results = []

    for prompt in batch_prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > max_input_tokens:
            truncated_tokens = tokens[-max_input_tokens:]
            new_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            results.append(new_prompt)
        else:
            results.append(prompt)
            
    return results


if __name__ == "__main__":
    args = parser.parse_args()
    # set seed global
    set_seed(args.seed)
    
    #load model
    login(token=HF_ACCESS_TOKEN)
    accelerator = Accelerator()
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        trust_remote_code=True
    )
    pipeline.model = accelerator.prepare(pipeline.model)

    # Fix missing pad_token (important for batch inference)
    # If the model has no pad_token, set pad_token = eos_token
    # This prevents generation errors when padding sequences in batch mode.
    if pipeline.tokenizer.pad_token is None:
        pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
        pipeline.model.config.pad_token_id = pipeline.tokenizer.eos_token_id

    # Optional: ensure tokenizer padding is on the right side
    # Right-side padding is generally safer for generation models.
    pipeline.tokenizer.padding_side = "right"

    # Load dataset
    experiment_datasets = load_from_disk(args.dataset_path)
    train_set = experiment_datasets[args.dataset_train_source]
    test_set = experiment_datasets["test"]
    
    
    if args.start is not None and args.end is not None:
        test_set = test_set.select(range(args.start,args.end))

    print(f"Loaded {len(test_set)} test samples")
    
    # Build prompts
    print("Building prompts...")
    prompts_ds = test_set.map(
        lambda x: build_prompt(x, train_set, max_shots=args.num_shots),
        num_proc=8,   # 8 core parallel, adjust according to CPU
        desc="Building prompts"
    )

    # Format prompts based on whether it is a chat model
    prompts = prompts_ds["prompt"]
    prompts = [
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|start_header_id|>assistant<|end_header_id|>
        """.format(system_prompt=system_prompt, user_prompt= prompt)
        if args.chat_model else
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ] 
        for prompt in prompts 
    ]

    # If pipeline fail because OOM
    if args.max_input_tokens is not None:
        batch_prompts = truncate_prompts(prompts, pipeline.tokenizer, max_input_tokens=args.max_input_tokens) 

    # Dry run
    if args.dry_run:
        print(prompts[0])
        print("Answer:", test_set[0]["answer"])
        sys.exit()
    
    # Batch inference
    results = []
    BATCH_SIZE = args.batch_size
    print(f"Running inference on {len(test_set)} samples (batch size={BATCH_SIZE})")

    for i in tqdm.tqdm(range(0, len(test_set), BATCH_SIZE), desc="Generating", ncols=100):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        batch_entries = test_set.select(range(i, min(i+BATCH_SIZE, len(test_set)))).to_list()

        outputs = pipeline(batch_prompts, max_new_tokens=256, batch_size=BATCH_SIZE)
        model_outputs = [ 
            extract_chat_model_output_text(out[0]["generated_text"]) if args.chat_model else out[0]["generated_text"][-1]['content'] 
            for out in outputs
        ]
        extracted_answers = [extract_answer(o) for o in model_outputs]
        
        for j, entry in enumerate(batch_entries):
            results.append({
                "id": entry["id"],
                "question": entry["question"],
                "answer": entry["answer_text"][0] if len(entry["answer_text"]) > 0 else "",
                "extracted_answer": extracted_answers[j],
                "model_output": model_outputs[j],
                "table_id": entry["table"]["id"],
            })
            
    if not args.dry_run:
        # Calculate metrics
        metrics = calculate_metrics(results)

        # Output file setup
        now = datetime.now()
        dt_string = now.strftime("%d_%H_%M")

        # Create model name safe for filesystem
        model_version = re.sub(r'[^a-zA-Z0-9._-]', '_', args.model)
        output_dir = f"indotapas-experiment/icl_approach/indohitab/outputs/{model_version}"
        os.makedirs(output_dir, exist_ok=True)
        
        
        output_path = (
            f"{output_dir}/output"
            f"{f'_start{args.start}_end{args.end}' if args.start is not None and args.end is not None else ''}"
            f"_{args.num_shots}shot"
            f"_seed{args.seed}"
            f"_{dt_string}.json"
        )
        # Save the output
        output_data = {
            "dataset_training_source": args.dataset_train_source,
            "num_shots": args.num_shots,
            "prompt":{
                "system": system_prompt,
                "shot_template":  TEMPLATE_FEWSHOT if args.num_shots >0 else TEMPLATE_ZEROSHOT
            },
            "metrics": metrics,
            "results": results
        }
        with open(output_path, "w") as fw:
            json.dump(
                output_data,
                fw,
                indent=2,
                ensure_ascii=False
            )
        
        # Print metrics
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Total Questions: {metrics['total']}")
        print(f"Exact Match (EM): {metrics['exact_match']:.2%}")
        print(f"F1 Score: {metrics['f1_score']:.2%}")
        print("="*50)
        
        fw.close()

    print("\nDone!")