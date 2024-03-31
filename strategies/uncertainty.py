import torch
from tqdm import tqdm




def transform(args, sample_pool_data, test_data, model, func):

    print('computing sentence uncertainty...')
    uncertainty = get_uncertainties(args, sample_pool_data[0], sample_pool_data[1], model, func)
    val, idx = uncertainty.topk(args.K)

    prompt = ''
    for i in idx.flip(0):
        question = sample_pool_data[0][i.item()]
        answer = sample_pool_data[1][i.item()]
        prompt += question + ' ' + answer + '. </s> '

    icl_test = []
    for x, y in zip(test_data[0], test_data[1]):
        icl_test.append(prompt + x)


    return icl_test, test_data[1], test_data[2], test_data[3]



def get_uncertainties(args, queries, labels, ref_model, func):
    inps = [x + ' ' + y for x, y in zip(queries, labels)]
    tokens = ref_model.tokenizer(inps)

    uncertainty_scores = []
    for data, mask in tqdm(zip(tokens['input_ids'], tokens['attention_mask']), total=len(queries)):
        inp = {'input_ids': torch.tensor([data]), 'attention_mask': torch.tensor([mask]), 'length':torch.tensor([len(data)])}
        dist = ref_model(inp, output_hidden_states=True, hidden_states_layers_to_output=(-1,), output_only_last_token_hidden_states=False)[1][0][-1].softmax(0)
        
        if func == 'bin':
            p_yes, p_no = get_yes_no_total_prob(args, dist)
            p_yes_norm, p_no_norm = p_yes/(p_yes + p_no), p_no/(p_yes + p_no)
            uncertainty = - p_yes_norm * torch.log(p_yes_norm) - p_no_norm * torch.log(p_no_norm)
        elif func == 'cat':
            uncertainty = -torch.sum(dist * torch.log(dist))
        uncertainty_scores.append(uncertainty)
        
    return torch.tensor(uncertainty_scores).cuda()



def get_yes_no_total_prob(args, dist):
    if args.model in ['opt']:
        yes_prob = dist[10932] + dist[4420] + dist[9904] + dist[3216] + dist[41010] + dist[32463]
        no_prob = dist[2362] + dist[117] + dist[3084] + dist[440] + dist[13449] + dist[8228]
    elif args.model in ['gptj']:
        yes_prob = dist[8505] + dist[3763] + dist[3363] + dist[5297] + dist[21560] + dist[43335]
        no_prob = dist[645] + dist[3919] + dist[1400] + dist[2949] + dist[8005] + dist[15285]
    elif args.model in ['llama','vicuna']:
        # YES token idx = 3582 (yes) & 3869 (▁Yes) & 4874 (▁yes) & 8241 (Yes) & 21143 (YES) & 22483 (▁YES)
        # NO token idx = 694 (▁no) & 1217 (no) & 1939 (▁No) & 3782 (No) & 6632 (NO) 11698 & (▁NO)
        yes_prob = dist[3582] + dist[3869] + dist[4874] + dist[8241] + dist[21143] + dist[22483]
        no_prob = dist[694] + dist[1217] + dist[1939] + dist[3782] + dist[6632] + dist[11698]
    else :
        raise NotImplementedError
    return yes_prob, no_prob