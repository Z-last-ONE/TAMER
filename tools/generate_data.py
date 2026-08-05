
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import FastICA, PCA
from tqdm import tqdm


def _resolve_path(data_path, file_name):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(data_path, file_name)


def feat_redistribution(
    data_path,
    file_name,
    redistribution_type1=True,
    redistribution_type2=True,
    n_components=100,
    epsilon=1e-5,
):
    input_path = _resolve_path(data_path, file_name)
    X = np.load(input_path, allow_pickle=False)
    if X.ndim != 2:
        raise ValueError(
            f"Text features must be a 2-D array, but {input_path!r} has "
            f"shape {X.shape}."
        )
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float32)

    print(f"Loaded text features: {input_path} {X.shape} {X.dtype}")

    if redistribution_type1:
        X_zca = X.T
        mu = np.mean(X_zca, axis=1, keepdims=True)
        X_centered = X_zca - mu
        cov_matrix = (
            (X_centered @ X_centered.T) / X_zca.shape[1]
            + epsilon * np.eye(X_zca.shape[0])
        )
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        D = eigenvectors
        Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + epsilon))
        Phi = D @ Lambda_inv_sqrt @ D.T
        Z = Phi @ X_centered

        path = os.path.join(data_path, "text_feat_z.npy")
        np.save(path, Z.T)
        print(f"Saved ZCA features: {path} {Z.T.shape} {Z.T.dtype}")

    if redistribution_type2:
        if n_components > min(X.shape):
            raise ValueError(
                f"n_components={n_components} cannot exceed the smaller "
                f"input dimension {min(X.shape)} for shape {X.shape}."
            )

        def apply_pca(features, components=100):
            pca = PCA(n_components=components)
            return pca.fit_transform(features)

        def apply_ica(features, components=100):
            ica = FastICA(
                n_components=components,
                random_state=42,
                whiten="unit-variance",
            )
            return ica.fit_transform(features)

        X_pca = apply_pca(X, n_components)
        X_ica = apply_ica(X_pca, n_components)

        path = os.path.join(data_path, "text_feat_p.npy")
        np.save(path, X_ica)
        print(f"Saved PCA-ICA features: {path} {X_ica.shape} {X_ica.dtype}")


def gen_item_matrix(all_edge, no_items, min_common_users=2):
    edge_dict = defaultdict(set)

    for edge in all_edge:
        user, item = edge
        edge_dict[int(item)].add(user)

    min_item = 0
    num_item = no_items
    item_graph_matrix = torch.zeros(num_item, num_item)
    key_list = sorted(edge_dict.keys())

    bar = tqdm(total=len(key_list), desc="Building item co-occurrence matrix")
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head + 1, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            inter_len = len(item_head.intersection(item_rear))
            if inter_len >= min_common_users:
                item_graph_matrix[head_key - min_item][rear_key - min_item] = inter_len
                item_graph_matrix[rear_key - min_item][head_key - min_item] = inter_len
    bar.close()

    return item_graph_matrix


def gen_ii(
    dataset_path,
    dataset_name,
    inter_file=None,
    topk=20,
    min_common_users=2,
):
    print("Data path:\t", dataset_path)
    uid_field = "userID"
    iid_field = "itemID"
    split_field = "x_label"

    if inter_file is None:
        inter_file = dataset_name.lower() + ".inter"
    inter_path = _resolve_path(dataset_path, inter_file)
    inter_df = pd.read_csv(inter_path, sep="\t")

    required_columns = {uid_field, iid_field, split_field}
    missing_columns = required_columns.difference(inter_df.columns)
    if missing_columns:
        raise ValueError(
            f"{inter_path!r} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    item_ids = np.sort(pd.unique(inter_df[iid_field]))
    expected_item_ids = np.arange(len(item_ids))
    if not np.array_equal(item_ids, expected_item_ids):
        raise ValueError(
            "TAMER preprocessing requires contiguous, zero-based itemID "
            f"values. Found {len(item_ids)} unique IDs with range "
            f"[{item_ids.min()}, {item_ids.max()}]."
        )

    num_item = len(item_ids)
    train_df = inter_df[inter_df[split_field] == 0].copy()
    train_data = train_df[[uid_field, iid_field]].to_numpy()
    print(
        f"Loaded interactions: {len(inter_df)} total, {len(train_df)} train, "
        f"{num_item} items"
    )

    item_graph_matrix = gen_item_matrix(
        train_data,
        num_item,
        min_common_users=min_common_users,
    )

    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    print(
        f"Keeping at most {topk} neighbors per item "
        f"(minimum shared users: {min_common_users})"
    )

    item_graph_dict = {}
    for i in tqdm(range(num_item), desc="Selecting top-k item neighbors"):
        item_num = len(torch.nonzero(item_graph_matrix[i]))
        keep_num = min(int(item_num), topk)
        item_i = torch.topk(item_graph_matrix[i], keep_num)
        edge_list_i = item_i.indices.numpy().tolist()
        edge_list_j = item_i.values.numpy().tolist()
        item_graph_dict[i] = [edge_list_i, edge_list_j]

    path = os.path.join(dataset_path, "item_graph_dict.npy")
    np.save(path, item_graph_dict, allow_pickle=True)
    print(f"Saved item graph dictionary: {path} ({len(item_graph_dict)} items)")


if __name__ == "__main__":
    data_file = "./data"
    dataset_name = "Baby"
    text_feature_file = "text_feat.npy"
    data_path = os.path.join(data_file, dataset_name)

    feat_redistribution(data_path, text_feature_file, True, True)
    gen_ii(data_path, dataset_name)
