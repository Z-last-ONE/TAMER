# coding: utf-8


"""
Utility functions
##########################
"""
import os

import numpy as np
import torch
import importlib
import datetime
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler


def build_non_zero_graph(adj, is_sparse=True, norm_type='sym'):
    device = adj.device
    nonzero_indices = adj.nonzero()
    i = nonzero_indices.T
    v = adj[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
    return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        edge_weight.masked_fill_(edge_weight <= 1e-7, 1e-7)
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def draw_t_sne(X, test_data):
    X = X.cpu()
    tst = test_data.dataset
    # tst = tst.loc[:99,['userID','itemID']]
    user_group = tst.df['userID'].drop_duplicates().values
    item_group = tst.df['itemID'].drop_duplicates().values + 19445
    user_group = X[user_group][:2000]
    item_group = X[item_group][:2000]

    colors = ["blue"] * 19445 + ["green"] * 7050

    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    plt.figure(figsize=(14, 10))
    plt.scatter(user_group[:, 0], user_group[:, 1], color="blue", s=4, label="Group 1")
    plt.scatter(item_group[:, 0], item_group[:, 1], color="green", s=4, label="Group 2")
    plt.title("T-SNE Embedding")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    plt.show()


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer():
    return getattr(importlib.import_module('common.trainer'), 'Trainer')


def init_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


############ LATTICE Utilities #########

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def decoder_loss_function(img_rep, de_txt, de_txt_c, de_txt_s, t):
    img = F.normalize(img_rep, dim=1)
    txt = F.normalize(de_txt, dim=1)
    txt_c = F.normalize(de_txt_c, dim=1)
    txt_s = F.normalize(de_txt_s, dim=1)
    pos_1 = torch.sum(img * txt_c, dim=1)
    pos_2 = torch.sum(img * txt, dim=1)
    neg_1 = torch.sum(img * txt_s)
    pos_1_h = torch.exp(pos_1 / t)
    pos_2_h = torch.exp(pos_2 / t)
    neg_1_h = torch.exp(neg_1 / t)
    loss_1 = -torch.mean(torch.log(pos_1_h / (pos_1_h + pos_2_h + neg_1_h) + 1e-24))
    loss_2 = -torch.mean(torch.log(pos_2_h / (pos_2_h + neg_1_h) + 1e-24))
    return loss_1 + loss_2


def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm


def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_tsne(embeddings, title="t-SNE with Angle Density"):
    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    data_2d = tsne.fit_transform(embeddings)

    # 归一化到 [0,1]，保证分布和原图相似
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_2d = scaler.fit_transform(data_2d)

    # 提取 t-SNE 降维后的坐标
    x_values = data_2d[:, 0]
    y_values = data_2d[:, 1]

    # 计算角度 (保持范围在 [-π, π])
    angles = np.arctan2(y_values - 0.5, x_values - 0.5)

    # 创建图像布局，使用 gridspec 调整子图大小比例
    fig = plt.figure(figsize=(5, 5))  # 调整整体大小，减少高度
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 仅保留散点图和密度图

    # 上方散点图（t-SNE 结果）
    ax1 = plt.subplot(gs[0])
    ax1.scatter(x_values, y_values, s=20, color='green', edgecolors='white', alpha=0.8)
    ax1.set_title(title)
    ax1.set_ylabel("Distribution")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([])  # 可选：隐藏X轴刻度

    # 下方密度图（角度分布）
    ax2 = plt.subplot(gs[1])
    sns.kdeplot(angles, ax=ax2, fill=True, color='green')
    ax2.set_xlabel("Angles")
    ax2.set_ylabel("Density")
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(0, 0.5)
    ax2.set_yticks([0, 0.25, 0.5])  # 精简Y轴刻度

    # 调整子图间距，减少留白
    plt.subplots_adjust(hspace=0.05)  # 进一步缩小间距

    # 使用紧凑布局
    plt.tight_layout()

    # 显示图像
    plt.show()


def draw_temp_pic(array, array2, num_x):
    # 截取前20×20的子数组
    sub_array = array[:num_x, :num_x]
    sub_array2 = array2[:num_x, :num_x]

    global_vmin = min(sub_array.min(), sub_array2.min())  # 0.1
    global_vmax = max(sub_array.max(), sub_array2.max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    start_hex = "#FDF5E6"
    end_hex = "#008B45"
    start_rgb = tuple(int(start_hex[i:i + 2], 16) for i in (1, 3, 5))  # 输出：(42, 157, 143)
    end_rgb = tuple(int(end_hex[i:i + 2], 16) for i in (1, 3, 5))  # 输出：(231, 111, 81)
    start_rgb_normalized = (start_rgb[0] / 255, start_rgb[1] / 255, start_rgb[2] / 255)
    end_rgb_normalized = (end_rgb[0] / 255, end_rgb[1] / 255, end_rgb[2] / 255)
    colors = [start_rgb_normalized, end_rgb_normalized]  # 从起始色到结束色
    cmap_name = "custom_gradient"
    n_bins = 256
    # 创建颜色映射对象
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # 绘制热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    im1 = ax1.imshow(sub_array, cmap=custom_cmap, vmin=global_vmin, vmax=global_vmax)
    im2 = ax2.imshow(sub_array2, cmap=custom_cmap, vmin=global_vmin, vmax=global_vmax)
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.02)
    plt.tight_layout()
    plt.show()
    # plt.figure(figsize=(8, 6))
    # heatmap = plt.imshow(sub_array, cmap=custom_cmap)
    # plt.colorbar(heatmap, label='')
    # plt.title('')
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.show()
