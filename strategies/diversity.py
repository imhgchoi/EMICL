import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np



def transform(args, sample_pool_data, test_data, model):

    print('computing sentence embeddings...')
    train_embs = get_embeddings(args, sample_pool_data[0], model)

    km = KMeans(n_clusters=args.K, n_init='auto').fit(train_embs.cpu())
    idx = return_indices(args.K, km, train_embs.cpu())

    prompt = ''
    for i in idx.flip(0):
        question = sample_pool_data[0][i.item()]
        answer = sample_pool_data[1][i.item()]
        prompt += question + ' ' + answer + '. </s> '

    icl_test = []
    for x, y in zip(test_data[0], test_data[1]):
        icl_test.append(prompt + x)


    return icl_test, test_data[1], test_data[2], test_data[3]



def return_indices(num_clusters, km, train_embs):
    all_data = [ i for i in range(train_embs.shape[0]) ]
    m_clusters = km.labels_.tolist()

    centers = np.array(km.cluster_centers_)

    closest_data = []
    for i in range(num_clusters):
        center_vec = centers[i]
        data_idx_within_i_cluster = [idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]

        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster), centers.shape[1]))
        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = train_embs[data_idx]
            one_cluster_tf_matrix[row_num] = one_row
        closest, _ = pairwise_distances_argmin_min(np.expand_dims(center_vec, 0), one_cluster_tf_matrix)
        closest_idx_in_one_cluster_tf_matrix = closest[0]
        closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
        data_id = all_data[closest_data_row_num]

        closest_data.append(data_id)

    closest_data = list(set(closest_data))
    assert len(closest_data) == num_clusters
    
    return closest_data



def get_embeddings(args, dataset, ref_model):
    dataset = [x[x.index("\n"):][2:-1] for x in dataset]
    lyr = -1
    tokens = ref_model.tokenizer(dataset)
    embeddings = []
    for data, mask in tqdm(zip(tokens['input_ids'], tokens['attention_mask']), total=len(dataset)):
        inp = {'input_ids': torch.tensor([data]), 'attention_mask': torch.tensor([mask]), 'length':torch.tensor([len(data)])}
        embs = ref_model(inp, output_hidden_states=True, hidden_states_layers_to_output=(lyr,), output_only_last_token_hidden_states=True)[0][0][0]
        embeddings.append(embs)
    return torch.stack(embeddings).cuda()