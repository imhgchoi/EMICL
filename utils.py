
# def load_model(args):
#     model_id = "upstage/Llama-2-70b-instruct-v2"
#     model_name = "Llama-2-70b-instruct-v2"

#     tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         load_in_8bit=True,
#         rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
#     )

#     pipeline = transformers.pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#     )

import json
import pandas as pd   
from models.llama import LLaMAWrapper

def load_model(args):
    if args.model == 'llama':
        model = LLaMAWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    return model

def load_data(args, split):
    if split == 'train' :
        data_dir = f'LLMForEM/data/{args.data}/{args.data}-train.jsonl' 
    elif split == 'test':
        data_dir = f'LLMForEM/data/{args.data}/{args.data}-valid.jsonl'

    with open(data_dir, 'r') as json_file:
        json_list = list(json_file)

    if args.debug :
        json_list = json_list[:50]

    queries, responses, entities1, entities2 = [], [], [], []
    for json_str in json_list:
        result = json.loads(json_str)

        query = result['messages'][0]['content']
        query = f'<s> [INST] {query}. Answer with Yes or No only. [/INST]'
        response = result['messages'][1]['content']
        entity1 = query.split("'")[-4].strip()
        entity2 = query.split("'")[-2].strip()

        queries.append(query)
        responses.append(response)
        entities1.append(entity1)
        entities2.append(entity2)

    return queries, responses, entities1, entities2

