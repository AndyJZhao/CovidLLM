import string

import hydra.utils
import torch as th

th.set_num_threads(1)

from bidict import bidict
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedShuffleSplit

import utils.basics as uf
from .prompt_tree import PromptTree
from utils.basics import logger
import numpy as np


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
                self.prompt.demo_qa(info=t.prompt, answer=t.label) for t in support_tree_list)
            demo_prompt = self.prompt.demo(demonstration=demonstration)
            return demo_prompt
        else:
            return ''

    def build_prompt_tree(self, id, supervised=False):
        # ! Center node graph
        label = self.df.iloc[id][self.cfg.target] if supervised else None
        prompt_tree = PromptTree(self.cfg, data=self, id=id,
                                 hierarchy=None, label=label, name_alias=self.cfg.tree_node_alias,
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
