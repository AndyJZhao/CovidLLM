import os
import string
from collections import Counter
from copy import deepcopy
import scipy.sparse as sp

import dgl
import hydra.utils
import numpy as np
import pandas as pd
import torch as th

th.set_num_threads(1)

import torch.nn.functional as F
from bidict import bidict
from dgl import PPR
from dgl import backend as dgl_F
from dgl import node_subgraph, to_bidirected, remove_self_loop
from easydict import EasyDict
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedShuffleSplit

import utils.basics as uf
import utils.pkg.graph_utils as g_utils
from utils.data.ppr import (
    calc_approximate_ppr_rank,
    topk_approximate_ppr_matrix,
    find_top_k_neighbors_within_khop_ego_subgraph,
)
from utils.data.preprocess import load_ogb_graph_structure_only
from utils.pkg.dict2xml import dict2xml
from utils.pkg.distributed import master_process_only, process_on_master_and_sync_by_pickle
from .prompt_tree import PromptTree
from utils.basics import logger
import numpy as np
from scipy.cluster.vq import kmeans, vq

CONTINUOUS_FIELDS = ["x", "tape_emb", "y", "a1x", "a2x", "a3x", 'a1y', 'a2y', 'a3y', "y_hat", "r"] + [f"h{i}" for i in
                                                                                                      range(10)]
LABEL_FIELDS = ['label_name', 'choice', 'y']


def get_stratified_subset_split(labels, label_subset, valid_ids, n_split_samples):
    # Subset stratified split from all labels
    # valid_ids: available ids
    ids_left = valid_ids
    split_ids = {}
    for split, n_samples in n_split_samples.items():
        if n_samples > 0:
            split_ids[split] = np.random.permutation(
                np.concatenate([ids_left[np.where(labels[ids_left] == l)[0][:n_samples]]
                                for l in label_subset
                                ]
                               )
            )
            ids_left = np.setdiff1d(ids_left, split_ids[split])
        else:
            split_ids[split] = []
    return split_ids


def initialize_label_and_choices(all_label_info, label_subset=None, use_alphabetical_choice=True):
    if label_subset is not None:
        label_info = all_label_info.iloc[label_subset].reset_index(drop=True)
    else:
        label_info = all_label_info
    if len(label_info) > 26 or (not use_alphabetical_choice):
        label_info["choice"] = [f"<c{i}>" for i in range(len(label_info))]
    else:  # Alphabetical
        label_info["choice"] = [string.ascii_uppercase[i] for i in range(len(label_info))]
    choice_to_label_name = bidict()
    choice_to_label_id = bidict()
    raw_label_id_to_label_id = bidict()
    label_info.rename(columns={'label_id': 'raw_label_id'}, inplace=True)
    label_info['label_id'] = np.arange(len(label_info))
    for i, row in label_info.iterrows():
        choice_to_label_name[row["choice"]] = row["label_name"]
        choice_to_label_id[row["choice"]] = row["label_id"]
        raw_label_id_to_label_id[row["raw_label_id"]] = row["label_id"]
    return label_info, choice_to_label_id, choice_to_label_name, raw_label_id_to_label_id


@master_process_only
def _prepare_ogb_cache(
        ogb_name,
        process_mode,
        raw_text_url,
        max_seq_len,
        n_labels,
        label_subset,
        sample_per_class,
        subset_class,
        graph_save_path,
        raw_data_path,
        info_file,
        processed_text_file,
        **kwargs,
):
    # ! Process Full Graph
    g, labels, split_idx = load_ogb_graph_structure_only(
        ogb_name, raw_data_path, save_path=graph_save_path
    )

    # Process and save supervision
    split_idx["val"] = split_idx.pop("valid")
    split_ids = {_: split_idx[_].numpy() for _ in ["train", "val", "test"]}
    if sample_per_class > 0 or subset_class != n_labels:
        # Top 4 frequently used classes are selected.
        g = to_bidirected(remove_self_loop(g), copy_ndata=True)
        g, split_ids = subset_graph(
            g, sample_per_class, split_ids, labels, label_subset
        )

    g_info = EasyDict(
        splits=split_ids,
        labels=labels,
        n_nodes=g.num_nodes(),
        IDs=np.arange(len(labels)),
    )  # Default Graph Info for FULL graph
    if sample_per_class > 0 or subset_class != n_labels:
        g_info.IDs = g.ndata["_ID"].numpy()
        g_info.labels = g_info.labels[g_info.IDs]

        # Resplit according to few or one-shot.
        valid_ids = np.concatenate([v for k, v in split_ids.items()])
        n_train_samples = 1  # TODO to be revisited
        n_test_samples = round(sample_per_class * 0.8)  # To be defined in configs.x
        n_split_samples = {
            "train": n_train_samples,
            "test": n_test_samples,
            "val": sample_per_class - n_train_samples - n_test_samples,
        }
        g_info.splits = get_stratified_subset_split(
            labels, label_subset, valid_ids, n_split_samples
        )
        g_info.n_labels = subset_class

    uf.pickle_save(g_info, info_file)
    del g

    if not os.path.exists(processed_text_file):
        if ogb_name == "ogbn-arxiv":
            from utils.data.preprocess import process_raw_arxiv

            process_raw_arxiv(
                labels,
                process_mode,
                ogb_name,
                raw_data_path,
                raw_text_url,
                max_seq_len,
                processed_text_file,
                _label_info=kwargs["_label_info"],
            )
        uf.logger.info(f"Text preprocessing finished")


def preprocess_ogb(ogb_name, process_mode, raw_data_path, raw_text_url, max_seq_len, info_file, processed_text_file,
                   n_labels, sample_per_class=1, demo: DictConfig = None, label_text=None, subset_class=None,
                   graph_save_path=None, additional_ndata=None, additional_text_data=None, **kwargs):
    subset_class = subset_class or n_labels
    if subset_class != n_labels:
        label_subset = kwargs["_label_order"][:subset_class]
    else:
        label_subset = np.arange(n_labels)
    _prepare_ogb_cache(
        ogb_name,
        process_mode,
        raw_text_url,
        max_seq_len,
        n_labels,
        label_subset,
        sample_per_class,
        subset_class,
        graph_save_path,
        raw_data_path,
        info_file,
        processed_text_file,
        **kwargs,
    )
    g_info = uf.pickle_load(info_file)

    g = load_ogb_graph_structure_only(ogb_name, raw_data_path, graph_save_path)[0]
    for ndata_field, data_file in additional_ndata.items():
        g.ndata[ndata_field] = th.load(data_file)

    # g = node_subgraph(g, g_info.IDs, relabel_nodes=False) # For subset
    g = to_bidirected(g, copy_ndata=True)
    text, all_label_info = uf.pickle_load(processed_text_file)
    # self.df = full_data.iloc[g_info.IDs].reset_index(drop=True)
    if 'tape' in additional_text_data:
        tape_df = pd.read_csv(additional_text_data['tape'], index_col=0)
        tape_df.rename(columns={'text': 'tape'}, inplace=True)
        text = pd.merge(tape_df, text, how="left", on="node_id")
        uf.logger.warning('Added TAPE feature.')
    # Create mask
    add_split_mask_to_graph(g, g_info.splits)
    return g, g_info, text, all_label_info, label_subset


def add_split_mask_to_graph(g, split_ids):
    for split in ["train", "val", "test"]:
        mask = th.zeros(g.num_nodes(), dtype=th.bool)
        mask[th.tensor(split_ids[split])] = True
        g.ndata[f"{split}_mask"] = mask


def subset_graph(
        g, sample_per_class, split_ids, labels, label_subset, ensure_sub_label=False
):
    # ! Subset labels first
    valid_ids = []
    for label in label_subset:
        subset_ids = np.where(labels == label)[0]
        subset_ids = np.intersect1d(subset_ids, th.where(g.in_degrees() > 0)[0].numpy())
        subset_ids = subset_ids[:sample_per_class] if sample_per_class else valid_ids
        valid_ids.append(subset_ids)
    # valid_ids = np.where(np.isin(labels, l-abel_subset))[0]
    valid_ids = np.concatenate(valid_ids)
    split_ids = {k: np.intersect1d(v, valid_ids) for k, v in split_ids.items()}

    # ! Subset graph
    if sample_per_class > 0 or label_subset != len(np.unique(labels)):
        subset_nodes = th.tensor(np.concatenate(list(split_ids.values())).astype(int))
        node_subset = g_utils.sample_nodes(g, subset_nodes, [-1])[0]
        if ensure_sub_label:
            node_subset = np.intersect1d(node_subset, valid_ids)
        g = node_subgraph(g, node_subset)

    return g, split_ids


def preprocess_dgl(data_cfg: DictConfig):
    dataset = hydra.utils.instantiate(data_cfg["_init_args"])
    g, labels = dataset[0], dataset[0].ndata["label"].numpy()
    if len(g.ndata['train_mask'].shape) > 1:
        # Multiple splits are provided, only the first split is selected.
        for s in ["train", "val", "test"]:
            g.ndata[f"{s}_mask"] = g.ndata[f"{s}_mask"][:, 0]
    split_ids = EasyDict(
        {
            s: np.random.permutation(np.where(g.ndata[f"{s}_mask"])[0])
            for s in ["train", "val", "test"]
        }
    )

    g_info = EasyDict(
        splits=split_ids,
        labels=labels,
        n_nodes=g.num_nodes(),
        IDs=np.arange(g.num_nodes()),
    )
    # ! Get text attribute
    # Get label information
    all_label_info = pd.DataFrame.from_dict({"label_id": [int(l) for l in data_cfg.label_name],
                                             "label_name": data_cfg.label_name.values(), })
    data = pd.DataFrame.from_dict({"label_id": labels})

    label_info, choice_to_label_id, choice_to_label_name, raw_label_id_to_label_id = initialize_label_and_choices(
        all_label_info, label_subset=None
    )
    data = pd.merge(data, label_info, how="left", on="label_id")
    data["text"] = data[data_cfg.df.mode]
    data["gold_choice"] = data.apply(
        lambda x: label_info.choice.get(x["label_id"], "Other Labels"), axis=1
    )
    data["pred_choice"] = np.nan

    label_lookup_funcs = (choice_to_label_id, choice_to_label_name)
    return g, g_info, data, label_info, label_lookup_funcs, dataset


def preprocess_explore_llm_on_graph(data_cfg: DictConfig):
    dataset = th.load(data_cfg.dataset_path, map_location="cpu")

    g = dgl.graph((dataset.edge_index[0], dataset.edge_index[1]))
    g.ndata["feat"] = dataset.x
    g.ndata["label"] = dataset.y
    for s in ["train", "val", "test"]:
        g.ndata[f"{s}_mask"] = dataset[f"{s}_masks"][0]

    labels = g.ndata["label"].numpy()
    split_ids = EasyDict({s: np.random.permutation(np.where(g.ndata[f"{s}_mask"])[0])
                          for s in ["train", "val", "test"]})
    g_info = EasyDict(
        splits=split_ids,
        labels=labels,
        n_nodes=g.num_nodes(),
        IDs=np.arange(g.num_nodes()),
    )

    all_label_info = pd.DataFrame.from_dict({
        "label_id": [l for l in range(len(dataset.label_names))],
        "label_name": dataset.label_names
    })
    data = pd.DataFrame.from_dict({"label_id": labels})

    label_info, choice_to_label_id, choice_to_label_name, raw_label_id_to_label_id = \
        initialize_label_and_choices(all_label_info, label_subset=None)

    data = pd.merge(data, label_info, how="left", on="label_id")
    data["text"] = dataset.raw_texts
    if (cutoff := data_cfg.get('text_cutoff')) is not None:
        data["text"] = data.apply(lambda x: ' '.join(x.df.split(' ')[:cutoff]), axis=1)
    data["gold_choice"] = data.apply(lambda x: label_info.choice.get(x["label_id"], "Other Labels"), axis=1)
    data["pred_choice"] = np.nan

    return g, g_info, data, label_info, choice_to_label_id, choice_to_label_name


class TopkPPRNeigbors(PPR):
    # ! NOTE THAT THIS IS ABANDONED
    # Neighbors with zero probability to connect is generated.
    def __call__(self, g, k):
        # ! Original PPR code from DGL
        # Step1: PPR diffusion
        # (α - 1) A
        device = g.device
        eweight = (self.alpha - 1) * g.edata.get(
            self.eweight_name, dgl_F.ones((g.num_edges(),), dgl_F.float32, device)
        )
        num_nodes = g.num_nodes()
        mat = dgl_F.zeros((num_nodes, num_nodes), dgl_F.float32, device)
        src, dst = g.edges()
        src, dst = dgl_F.astype(src, dgl_F.int64), dgl_F.astype(dst, dgl_F.int64)
        mat[dst, src] = eweight
        # I_n + (α - 1) A
        nids = dgl_F.astype(g.nodes(), dgl_F.int64)
        mat[nids, nids] = mat[nids, nids] + 1
        # α (I_n + (α - 1) A)^-1
        diff_mat = self.alpha * dgl_F.inverse(mat)

        # ! Modified
        # Remove self from ranked neighbors
        diff_mat.diagonal().fill_(-float("inf"))
        # Select topK indices of PageRank
        topk_values, topk_neighbor = th.topk(diff_mat, k, dim=1)

        # ! Analysis
        # nb_rank = {}
        # for _id in g.nodes():
        #     nb_rank[_id] = th.cat([th.where(th.argsort(diff_mat[_id, :], descending=True) == n)[0]
        #                            for n in g.successors(_id)]).numpy().tolist()
        # sub_nb_rank = {k: v for k, v in nb_rank.items() if k < 10}
        # print(
        #     f'alpha = {self.alpha}, nb_rank = {sub_nb_rank}')

        result = topk_neighbor.cpu().numpy()
        return result


def generate_few_shot_split(n_labels, g, split_ids, n_demo_per_class):
    demo_ids = []
    labels = th.tensor(g.ndata['label'])  # assuming the label is stored in graph ndata

    for l in np.arange(n_labels):
        label_nodes = split_ids['train'][np.where(labels[split_ids['train']] == l)[0]]
        demo_id = label_nodes[th.argsort(g.out_degrees(label_nodes))[-n_demo_per_class:]]
        demo_id = demo_id.reshape(-1).tolist()
        demo_ids.extend(demo_id)

    all_ids = np.concatenate([split_ids['train'], split_ids['val'], split_ids['test']])
    remaining_ids = list(set(all_ids) - set(demo_ids))
    remaining_labels = labels[remaining_ids]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    try:
        for val_index, test_index in sss.split(remaining_ids, remaining_labels):
            new_val_ids = np.array(remaining_ids)[val_index]
            new_test_ids = np.array(remaining_ids)[test_index]
    except:  # If failed, use random split
        permuted = np.random.permutation(remaining_ids)
        new_val_ids, new_test_ids = permuted[:len(permuted) // 2], permuted[len(permuted) // 2:]
    new_split_ids = {
        'train': np.array(demo_ids),
        'val': new_val_ids,
        'test': new_test_ids
    }

    return new_split_ids


class CovidData:
    @uf.time_logger("dataset initialization")
    def __init__(self, cfg: DictConfig):  # Process split settings, e.g. -1/2 means first split
        self.cfg = cfg
        # ! Initialize Data Related
        self.raw_data = raw_data = uf.pickle_load(cfg.data.raw_data_file)
        self.df = df = raw_data.merged_dynamic
        self.split_ids = splits = raw_data.splits
        self.label_info = label_info = raw_data.label_info
        logger.info(f'Loaded meta information of {len(raw_data.static)} states')
        logger.info(f'Loaded COVID data, {len(df)} weeks in total')

        # ! Splits
        for split, split_id in splits.items():
            split_df = df.iloc[split_id]
            logger.info(
                f'{split.upper()} set ({len(split_df)}): from {split_df.Week_start.min()} to '
                f'{split_df.Week_start.max()}')

        # ! Get label names
        self.label_names = str(self.label_info.label_name.tolist())
        if self.cfg.remove_quotation:
            self.label_names = self.label_names.replace('"', "").replace("'", "")

        # ! Initialize Prompt Related
        # Initialize classification prompt
        assert (col := f"label_{cfg.data.label_text}") in self.label_info.columns, "Unknown classification prompt mode."
        cfg.data.label_description = "[" + ", ".join(
            f'{_.label_token}: {_[col]}' for i, _ in self.label_info.iterrows()) + "]"

        self.prompt = hydra.utils.instantiate(cfg.prompt)
        uf.logger.info(self.prompt.human)

        return

    def __getitem__(self, item):
        return self.df.iloc[item]

    def build_demo_prompt(self, support_tree_list):
        if len(support_tree_list) > 0:
            demo_cfg = self.cfg.demo
            sep = '\n' * demo_cfg.n_separators
            demonstration = sep.join(
                self.prompt.demo_qa(graph_info=t.prompt, answer=t.label) for t in support_tree_list)
            demo_prompt = self.prompt.demo(demonstration=demonstration)
            return demo_prompt
        else:
            return ''

    def build_prompt_tree(self, id, supervised=False):
        # ! Center node graph
        hierarchy = self.cfg.tree_hierarchy.split('.')
        label = self.df.iloc[id][self.cfg.target] if supervised else None
        prompt_tree = PromptTree(self.cfg, data=self, id=id,
                                 hierarchy=hierarchy, label=label, name_alias=self.cfg.tree_node_alias,
                                 style=self.cfg.prompt.style)
        return prompt_tree

    def select_demo(self, select_method, node_id):
        if (n_demos := self.cfg.demo.n_samples) <= 0:
            return []
        one_fixed_sample_for_each_class_funcs = ['first', 'max_degree']
        if select_method in one_fixed_sample_for_each_class_funcs:
            n_demo_per_class = max(n_demos // self.n_labels, 1)
            # Overwrite n_demos
            if select_method == 'first':  # Initialize if haven't
                demo_ids = np.concatenate(
                    [self.split_ids.train[np.where(self.labels[self.split_ids.train] == l)[0][:n_demo_per_class]] for l
                     in
                     np.arange(self.n_labels)])
            else:
                raise ValueError(f'Unsupported demo selection method {select_method}')
            return demo_ids
