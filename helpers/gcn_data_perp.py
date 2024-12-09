import numpy as np
from sklearn.model_selection import StratifiedKFold
import dill
import os
import pandas as pd
import torch
from torch_geometric.utils import degree
from torch_geometric.data import Data

# Nested Stratified 5-Fold cross validation
def get_fold_indices(args,labels):
    out_c = 0
    fold_idx = dict()
    skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    data = np.random.rand(len(labels),10)

    for train_idx1, test_idx in skf1.split(data,labels):
        out_c +=1
        in_c = 0
        
        train_data1 = [data[index] for index in train_idx1]
        labels1 = labels[train_idx1]
        
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for train_idx2, val_idx1 in skf2.split(train_data1,labels1):
            in_c +=1
            fold_idx['outer{}_inner{}'.format(out_c,in_c)]={}
            train_idx = train_idx1[train_idx2]
            val_idx = train_idx1[val_idx1]
            np.random.shuffle(train_idx) # shuffle train indices
            
            fold_idx['outer{}_inner{}'.format(out_c,in_c)]['train']=train_idx
            fold_idx['outer{}_inner{}'.format(out_c,in_c)]['test']=test_idx
            fold_idx['outer{}_inner{}'.format(out_c,in_c)]['val']=val_idx
        
    return fold_idx

def get_fold_indices1(args, labels):
    out_c = 0
    fold_idx = {}
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    data = np.random.rand(len(labels), 10)

    for train_idx, test_idx in skf_outer.split(data, labels):
        out_c += 1
        fold_idx['outer{}'.format(out_c)] = {}

        # Splitting train indices into train and validation sets
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for train_inner_idx, val_idx in skf_inner.split(data[train_idx], labels[train_idx]):
            train_idx_inner = train_idx[train_inner_idx]
            break  # Only using the first split for validation

        # Shuffle train indices
        np.random.shuffle(train_idx_inner)

        fold_idx['outer{}'.format(out_c)]['train'] = train_idx_inner
        fold_idx['outer{}'.format(out_c)]['test'] = test_idx
        fold_idx['outer{}'.format(out_c)]['val'] = val_idx

    return fold_idx

def get_fold_data(data,labels,indices,name,out_loop,in_loop):
    n_channels = data.shape[1]
    data = [ data[i] for i in indices['outer{}_inner{}'.format(out_loop,in_loop)][name]]
    data = data[indices['outer{}_inner{}'.format(out_loop,in_loop)][name],:]
    
    # extract upper triangle elements
    data_up = np.zeros((len(data),int(n_channels*(n_channels-1)/2))) # (N*(N-1))/2 each
    for i in range(len(data)):
        A = data[i]
        data_up[i] = A[np.triu_indices_from(A, k=1)]
    labels = [ labels[i] for i in indices['outer{}_inner{}'.format(out_loop,in_loop)][name]]
    return data_up, np.array(labels)

def build_graph_data_mean(args):
    proc_data_path = args.PROC_DATA_DIR
    fc_mats = dill.load(open(proc_data_path, "rb"))[0]
    patient_ids = dill.load(open(proc_data_path, "rb"))[2] # load the patient IDs (170 subjects)
    df = pd.read_excel('data/eeg_data_all_good-85_poor-85.xlsx',dtype={'CPC':'int', 'Patient':'string'}) #202 subjects before removing those with invalid data
    df = df[df['Patient'].isin(patient_ids)] # restricted to 170 subjects according to the patients_ids
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    labels = np.array([0 if item=='Good' else 1 for item in df['Outcome']]) # convert good/poor labels to binary (0: good, 1: poor)
    # CPC = df['CPC'].values # You can also use gcn to predict the CPC
    
    n_subjects = len(fc_mats)
    num_nodes = fc_mats[0].shape[0]
    node_features = []
    phenotypic_data = np.zeros([num_nodes * n_subjects, 3], dtype=np.float32)

    for i, fc_i in enumerate(fc_mats):
        # print(f'processing subject {i}')
        node_features.append(fc_i) # use FC as node features

        phenotypic_data[num_nodes * i:num_nodes * (i + 1), 0] = float(ord(df['Hospital'][i]))
        phenotypic_data[num_nodes * i:num_nodes * (i + 1), 1] = float(df['Age'][i])
        phenotypic_data[num_nodes * i:num_nodes * (i + 1), 2] = 1.0 if df['Sex'][i]=='Male' else 2.0 # Male:1   Female:2

    # Normalize the networks
    norm_networks = fc_mats.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(len(norm_networks)):
            np.fill_diagonal(norm_networks[i], 0) # remove the self-connections for the sake of normalization
            norm_networks[i] = np.nan_to_num(np.arctanh(norm_networks[i]))
            np.fill_diagonal(norm_networks[i], 1) # put back the self-connections as some nodes are disconnected in the graph

    pd_dict = {'HOSPITAL': phenotypic_data[:, 0],
               'AGE_AT_SCAN': phenotypic_data[:, 1],
               'SEX': phenotypic_data[:, 2]}

    return node_features, norm_networks, labels, pd_dict

def build_graph_data_sd(args):
    proc_data_path = args.PROC_DATA_DIR
    fc_mats = dill.load(open(proc_data_path, "rb"))[1]
    patient_ids = dill.load(open(proc_data_path, "rb"))[2] # load the patient IDs (170 subjects)
    df = pd.read_excel('data/eeg_data_all_good-85_poor-85.xlsx',dtype={'CPC':'int', 'Patient':'string'}) #202 subjects before removing those with invalid data
    df = df[df['Patient'].isin(patient_ids)] # restricted to 170 subjects according to the patients_ids
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    labels = np.array([0 if item=='Good' else 1 for item in df['Outcome']]) # convert good/poor labels to binary (0: good, 1: poor)
    # CPC = df['CPC'].values # You can also use gcn to predict the CPC
    
    n_subjects = len(fc_mats)
    num_nodes = fc_mats[0].shape[0]
    node_features = []
    phenotypic_data = np.zeros([num_nodes * n_subjects, 3], dtype=np.float32)

    for i, fc_i in enumerate(fc_mats):
        # print(f'processing subject {i}')
        node_features.append(fc_i) # use FC as node features

        phenotypic_data[num_nodes * i:num_nodes * (i + 1), 0] = float(ord(df['Hospital'][i]))
        phenotypic_data[num_nodes * i:num_nodes * (i + 1), 1] = float(df['Age'][i])
        phenotypic_data[num_nodes * i:num_nodes * (i + 1), 2] = 1.0 if df['Sex'][i]=='Male' else 2.0 # Male:1   Female:2

    # Normalize the networks
    norm_networks = fc_mats.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(len(norm_networks)):
            np.fill_diagonal(norm_networks[i], 0) # remove the self-connections for the sake of normalization
            norm_networks[i] = np.nan_to_num(np.arctanh(norm_networks[i]))
            np.fill_diagonal(norm_networks[i], 1) # put back the self-connections as some nodes are disconnected in the graph

    pd_dict = {'HOSPITAL': phenotypic_data[:, 0],
               'AGE_AT_SCAN': phenotypic_data[:, 1],
               'SEX': phenotypic_data[:, 2]}

    return node_features, norm_networks, labels, pd_dict


def build_signed_knn_graph(Corr, k=5):
    # Set the diagonal to zero to avoid self-loops
    np.fill_diagonal(Corr, 0)  
    # Specify the number of neighbors (k) for the KNN graph
    k_value = k
    # Find the top-k neighbors for each node based on magnitudes of correlation coefficients
    knn_adj_matrix = np.zeros_like(Corr)
    for i in range(Corr.shape[0]):
        # Get indices of the top-k neighbors for node i based on magnitudes
        top_k_indices_pos = np.argsort(Corr[i, :])[-k_value:]
        top_k_indices_neg = np.argsort(-Corr[i, :])[-k_value:]
        # Set the corresponding entries in the signed and weighted adjacency matrix
        knn_adj_matrix[i, top_k_indices_pos] = Corr[i, top_k_indices_pos]
        knn_adj_matrix[i, top_k_indices_neg] = Corr[i, top_k_indices_neg]
        # Ensure symmetry by adding the same edges for node j
        knn_adj_matrix[top_k_indices_pos, i] = Corr[i, top_k_indices_pos]
        knn_adj_matrix[top_k_indices_neg, i] = Corr[i, top_k_indices_neg]
    # Ensure symmetry
    knn_adj_matrix = 0.5 * (knn_adj_matrix + knn_adj_matrix.T)
    np.fill_diagonal(knn_adj_matrix, 1)
    assert np.count_nonzero(knn_adj_matrix) % 2 == 0
    assert np.allclose(knn_adj_matrix, knn_adj_matrix.T), "The array is not symmetric."
    try:
        assert np.sum(np.abs(knn_adj_matrix), axis=0).all()  # There are 0-in-degree nodes in the graph
    except: 
        input("1-Node with zero degree:\n")   
    
    return knn_adj_matrix

def build_knn_graph(Corr, k=5):
    # knn_graph = kneighbors_graph(Corr, n_neighbors=k, mode='connectivity', include_self=True).toarray()
    # Set the diagonal to zero to avoid self-loops
    np.fill_diagonal(Corr, 0)  
    # Specify the number of neighbors (k) for the KNN graph
    k_value = k
    # Find the top-k neighbors for each node based on magnitudes of correlation coefficients
    knn_adj_matrix = np.zeros_like(Corr)
    for i in range(Corr.shape[0]):
        # Get indices of the top-k neighbors for node i based on magnitudes
        top_k_indices = np.argsort(np.abs(Corr[i, :]))[-k_value:]
        # Set the corresponding entries in the signed and weighted adjacency matrix
        knn_adj_matrix[i, top_k_indices] = Corr[i, top_k_indices]
        # Ensure symmetry by adding the same edges for node j
        knn_adj_matrix[top_k_indices, i] = Corr[i, top_k_indices]

    np.fill_diagonal(knn_adj_matrix, 1)
    assert np.count_nonzero(knn_adj_matrix) % 2 == 0
    assert np.allclose(knn_adj_matrix, knn_adj_matrix.T),"The array is not symmetric."
    try:
        assert np.sum(np.abs(knn_adj_matrix),axis=0).all() # There are 0-in-degree nodes in the graph
    except: 
            input("1-Node with zero degree:\n")   
    
    return knn_adj_matrix


def get_graph_inputs(node_ftr, graph_net, labels, cv_idx, knn_k):
    """
    get edge indices and edge weights/attributes for ev-mgcn 
    Prepare graph data for PyTorch Geometric.
    Parameters:
    - node_ftr (numpy.ndarray): Node features.
    - graph_net (list of numpy.ndarray): Graph adjacency matrices.
    - labels (numpy.ndarray): Graph labels.
    - cv_idx (list): Cross-validation indices.
    - knn_k (int): Number of nearest neighbors for graph construction.

    Returns:
    - list of Data: List of PyTorch Geometric Data objects.
    """
    dataset_list = []
    for idx in cv_idx:
        # print(f'processing graph {idx}')
        aff_adj = build_knn_graph(graph_net[idx], k=knn_k) 
        np.fill_diagonal(aff_adj, 1) # TODO: Do we need self-loops?
        # create cross-node edge attributes and edge_index (adj) data
        n = aff_adj.shape[0] 
        # num_edge = n*(1+n)//2 - n   # without self connections
        num_edge = n*(1+n)//2# - n  # including self connections
        edge_index = np.zeros([2, num_edge], dtype=np.int64) 
        aff_score = np.zeros(num_edge, dtype=np.float32) 
        # prepare edge_index and cross-node edge_attr
        flatten_ind = 0
        # for i in range(n):
        #     for j in range(i+1, n): # without self-connections (upper triangle only)
        #         edge_index[:,flatten_ind] = [i,j]
        #         aff_score[flatten_ind] = aff_adj[i][j]  
        #         flatten_ind +=1
        for i in range(n):
            for j in range(i,n): # including self connections (upper triangle only)
                edge_index[:,flatten_ind] = [i,j]
                aff_score[flatten_ind] = aff_adj[i][j]  
                flatten_ind +=1
 
        assert flatten_ind == num_edge, "Error in computing edge input"
        # Compute the degree of each node
        deg = degree(torch.tensor(edge_index[0]))
        # Use assert to check if there are zero-degree nodes
        assert torch.min(deg) > 0, "Graph contains zero-degree nodes"

        edge_index_coo = torch.tensor(edge_index, dtype = torch.long)
        feature_matrix = torch.tensor(node_ftr[idx], dtype = torch.float32)
        # Create PyTorch Geometric Data object
        graph_data = Data(x = feature_matrix, edge_index=edge_index_coo, y=torch.tensor(labels[idx]))
        dataset_list.append(graph_data)

    return dataset_list
