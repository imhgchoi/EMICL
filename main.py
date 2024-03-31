import numpy as np
import torch

import argparse
import math, copy
from datetime import datetime

# from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from transformers import (
#     set_seed,
#     DataCollatorForLanguageModeling,
#     TrainingArguments,
#     Trainer
# )

from utils import *


def map_text_to_action(outputs):
    preds = []
    for output in outputs :
        if output in [' Yes.', ' Yes', 'Yes.', 'Yes', ' yes.', ' yes', 'yes', 'yes.']:
            preds.append(1)
        elif output in [' No.', ' No', 'No.', 'No', ' no.', ' no', 'no', 'no.']:
            preds.append(0)
        else :
            preds.append(-1)
    return preds


def evaluate(args, model, te_data):
    
    predictions = []
    for query, label, ent1, ent2 in tqdm(zip(te_data[0], te_data[1], te_data[2], te_data[3]), total=len(te_data[0])):
        response, logit = model.generate(args, query, verbose=False)

        if args.verbose :
            print(':::QUERY:::\n'+query)
            print(':::RESPONSE:::\n'+response)
            
        actions = map_text_to_action([response])
        predictions.append(actions[0])
    labels = map_text_to_action(te_data[1])

    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    accuracy = accuracy_score(labels, predictions)

    print(f"F1 = {f1}")
    print(f"Pr = {precision}")
    print(f"Re = {recall}")
    print(f"Ac = {accuracy}")
    

def main(args):
    model = load_model(args)
    
    sample_pool_data = load_data(args, 'train')
    test_data = load_data(args, 'test')

    if args.strategy == 'similarity' :
        from strategies.similarity import transform
        test_data = transform(args, sample_pool_data, test_data, model)
    elif args.strategy == 'uncertainty' :
        from strategies.uncertainty import transform
        test_data = transform(args, sample_pool_data, test_data, model, args.uncertainty_func)
    elif args.strategy == 'diversity' :
        from strategies.diversity import transform
        test_data = transform(args, sample_pool_data, test_data, model)
        
    evaluate(args, model, test_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str)

    # data args
    parser.add_argument('--data', type=str, default='abt-buy')
    parser.add_argument('--max_input_len', type=int, default=256)
    parser.add_argument('--debug', action='store_true')

    # model args
    parser.add_argument('--model', type=str, default="llama")
    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/")
    # parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--verbose', action='store_true')

    # ICL args
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--midlayer_for_sim', action='store_true')
    parser.add_argument('--penultlayer_for_sim', action='store_true')
    parser.add_argument('--choose_certain', action='store_true')
    parser.add_argument('--uncertainty_func', type=str, default='bin', choices=['bin','cat'])

    # train args
    # parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default="")

    return parser.parse_args()

if __name__ == '__main__' :
    args = parse_args()
    main(args)