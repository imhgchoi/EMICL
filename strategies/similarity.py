
import torch
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial import distance_matrix

def jaccard_similarity(set1, set2):
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    return intersection / union

def jaccard_distance(set1, set2):
    return 1 - jaccard_similarity(set1, set2)


def transform(args, sample_pool_data, test_data, model, func):

    print('computing sentence embeddings...')

    train_embs = get_embeddings(args, sample_pool_data[0], model)
    test_embs = get_embeddings(args, test_data[0], model)

    if func == 'mat':
        sim = test_embs @ train_embs.T
        val, idx = sim.topk(args.K)
    elif func == 'cosine':
        sim = cosine_similarity(test_embs.cpu().numpy(), train_embs.cpu().numpy())
        sim_tensor = torch.tensor(sim)
        val, idx = sim_tensor.topk(args.K, dim=1)
    elif func == 'l2':
        sim = distance_matrix(test_embs.cpu().numpy(), train_embs.cpu().numpy())
        sim = 1 / (1 + sim)
        sim_tensor = torch.tensor(sim)
        val, idx = (-sim_tensor).topk(args.K, dim=1)
    
    icl_test = []
    for x, y, ex_idx in zip(test_data[0], test_data[1], idx.cpu()):
        prompt = ''
        for i in ex_idx.flip(0):
            question = sample_pool_data[0][i.item()]
            answer = sample_pool_data[1][i.item()]
            prompt += question + ' ' + answer + '. </s> '
        
        icl_test.append(prompt + x)


    return icl_test, test_data[1], test_data[2], test_data[3]


def get_embeddings(args, dataset, ref_model):
    dataset = [x[x.index("\n"):][2:-1] for x in dataset]
    if args.midlayer_for_sim :
        lyr = -0.5
    elif args.penultlayer_for_sim :
        lyr = -2
    else :
        lyr = -1
    tokens = ref_model.tokenizer(dataset)
    embeddings = []
    for data, mask in tqdm(zip(tokens['input_ids'], tokens['attention_mask']), total=len(dataset)):
        inp = {'input_ids': torch.tensor([data]), 'attention_mask': torch.tensor([mask]), 'length':torch.tensor([len(data)])}
        embs = ref_model(inp, output_hidden_states=True, hidden_states_layers_to_output=(lyr,), output_only_last_token_hidden_states=True)[0][0][0]
        embeddings.append(embs)
    return torch.stack(embeddings).cuda()
