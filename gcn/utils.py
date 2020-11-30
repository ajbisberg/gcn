import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse import csr_matrix,lil_matrix
import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from math import floor
seed = 123


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def adj_from_features(teams,cont_cols,bin_cols,use_delta=False):
    feat_cols = cont_cols + bin_cols
    ordered_teams = list(teams.sort_values(['date','team'])['team'].unique())
    ordered_games = list(teams.sort_values(['date','gameid'])['gameid'].unique())
    teams['adj_team_idx'] = teams['team'].apply(lambda team: ordered_teams.index(team))
    teams['adj_game_idx'] = teams['gameid'].apply(lambda game: ordered_games.index(game))
    srt_teams = teams.sort_values(['adj_team_idx','adj_game_idx']).reset_index()

    labels = [[0,0] for r in range(len(srt_teams))]
    label_idx = []
    ul_idx = []
    curr_team = ordered_teams[0]

    features = []

    adj = np.zeros([len(teams),len(teams)])
    team_name = ''
    for i,row in srt_teams.iterrows():
        if row['team'] == team_name:
            # related to next game
            adj[i,i-1] = 1
        else:
            team_name = row['team']
        opp = srt_teams[(srt_teams['adj_game_idx'] == row['adj_game_idx']) & (srt_teams['team'] != row['team'])]
        opp_idx = opp.index[0]
        if use_delta:
          delta_feats = np.subtract(row[cont_cols].values, opp[cont_cols].values[0])
          features.append(np.append(delta_feats,row[bin_cols].values))
        adj[i,opp_idx] = 1 

        if i < len(srt_teams) -1:
            if curr_team == srt_teams.iloc[i+1]['team']:
                labels[i] = [1,0] if srt_teams.iloc[i+1]['result'] == 1 else [0,1]
                label_idx.append(i)
            else:
                curr_team = srt_teams.iloc[i+1]['team']
                ul_idx.append(i)
        if i == len(srt_teams)-1:
            ul_idx.append(i)

    if not use_delta:
      features = srt_teams[feat_cols].values
  
    return (
        adj, 
        np.array(features),
        np.array(labels), 
        np.array(label_idx)
    )

def train_val_split_idx(train_pct,val_pct,label_idx):
    tr = floor(len(label_idx)*train_pct)
    val = floor(len(label_idx)*val_pct)
    np.random.seed(seed)
    tr_idx = np.random.choice(label_idx,tr,replace=False)
    remaining_idx = list(set(label_idx) - set(tr_idx))
    val_idx = np.random.choice(remaining_idx,val,replace=False)
    test_idx = np.array(list(set(remaining_idx) - set(val_idx)))
    return tr_idx, val_idx, test_idx

def load_data_lol(dataset_paths,data_type):
    """
    Loads input data from OE CSV 

    :param dataset_path: path to dataset
    :return: All data input files loaded (as well the training/test data).
    """
    print(" \n Begin processing data... \n")
    lol = pd.DataFrame()
    for dp in dataset_paths:
      lol = lol.append(pd.read_csv(dp))
    info = ['datacompleteness','gameid','year','split','playoffs','patch','date','league','team','gamelength']
    small_info = ['date','gameid','team',]
    obj = ['side','dragons','heralds','barons','towers','inhibitors',] # 'firstdragon','firstmidtower','firsttothreetowers','elders', 'elementaldrakes','firsttower','firstbaron'
    farm = ['monsterkills', 'monsterkillsenemyjungle', 'monsterkillsownjungle',] #'total cs','minionkills'
    goldxp = ['golddiffat10', 'golddiffat15','totalgold','earnedgold','xpdiffat10','xpdiffat15'] # 
    fights = ['firstblood','kills','assists','deaths','doublekills','triplekills','quadrakills','pentakills', 'damagetochampions', 'damagemitigatedperminute','damagetakenperminute' ]
    vision = ['controlwardsbought', 'visionscore', 'wardskilled', 'wardsplaced',] 
    res = ['result',]

    binary_cols = ['side','firstblood','result'] # 'firstdragon','firstbaron','firsttower', 'firsttothreetowers', 'firstmidtower',
    data_cols = obj+farm+goldxp+fights+vision
    cont_cols = [c for c in data_cols if c not in binary_cols]

    teams = lol[lol['player'].isna()][info+data_cols+res]
    teams = teams.dropna(subset=['doublekills','triplekills','quadrakills','pentakills',])
    teams['date'] = pd.to_datetime(teams['date'])
    teams['side'] = [1 if s=='Red' else 0 for s in teams['side']]

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    teams[data_cols] = imputer.fit_transform(teams[data_cols].astype('float32'))
    ss = StandardScaler()
    # teams[data_cols] = ss.fit_transform(teams[data_cols])

    # separate datasets for train/test/val
    train = teams[teams['league'].isin(['LPL'])] # ,'LCK','LEC','LCS']
    val = teams[teams['league'].isin(['LCK'])]
    test = teams[teams['league'].isin(['LCS'])]
    # adj, features, labels, label_idx = adj_from_features(teams,data_cols+res)
    use_delta = True if data_type == 'delta' else False
    train_adj, train_features, train_labels, train_label_idx = adj_from_features(train,cont_cols,binary_cols,use_delta)
    val_adj, val_features, val_labels, val_label_idx = adj_from_features(val,cont_cols,binary_cols,use_delta)
    test_adj, test_features, test_labels, test_label_idx = adj_from_features(test,cont_cols,binary_cols,use_delta)
    #
    train_features[:,0:len(cont_cols)] = ss.fit_transform(train_features[:,0:len(cont_cols)])
    val_features[:,0:len(cont_cols)] = ss.fit_transform(val_features[:,0:len(cont_cols)])
    test_features[:,0:len(cont_cols)] = ss.fit_transform(test_features[:,0:len(cont_cols)])
    #
    l_tr = train_adj.shape[0]
    l_val = val_adj.shape[0]
    l_test = test_adj.shape[0]
    all_data_len = l_tr + l_val + l_test
    adj = np.zeros((all_data_len,all_data_len))
    adj[:l_tr,:l_tr] = train_adj
    adj[l_tr:l_tr+l_val,l_tr:l_tr+l_val] = val_adj
    adj[l_tr+l_val:all_data_len,l_tr+l_val:all_data_len] = test_adj
    
    features = np.vstack((train_features,val_features,test_features))
    labels = np.vstack((train_labels,val_labels,test_labels))

    train_mask = sample_mask(train_label_idx, len(features))
    val_mask = sample_mask(np.add(val_label_idx,l_tr), len(features))
    test_mask = sample_mask(np.add(test_label_idx,l_tr+l_val), len(features))

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    print(" \n Data processed \n")

    return csr_matrix(adj,dtype=np.float32), lil_matrix(features,dtype=np.float32), y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    print(
        'the feature vectors of both labeled and unlabeled training instances : {}\n'.format(allx.shape),
        'the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object :{}\n'.format(x.shape),
        'the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object : {}\n'.format(tx.shape))
    print(
        'the labels for instances in ind.dataset_str.allx as numpy.ndarray object : {}\n'.format(ally.shape),
        'he one-hot labels of the test instances as numpy.ndarray object : {}\n'.format(ty.shape)
    )
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    
    # stack training and test features
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # stack training and test labels
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    # print(adj.shape, features.shape, labels.shape)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    
    print('train ',idx_train)
    print('val ',idx_val)
    print('test ',min(idx_test),max(idx_test))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
