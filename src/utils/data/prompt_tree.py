import dgl
from omegaconf import DictConfig
import pandas as pd
import json
from copy import deepcopy
from utils.pkg.dict2xml import dict2xml
import re


class PromptTree:
    def __init__(self, data, id, hierarchy, name_alias, style='xml', label=None):
        self.df, self.static_df = data.df, data.static_df
        self.style = style
        self.hierarchy = hierarchy
        self.label = label
        self.tree_dict = self.df.iloc[id].to_dict()
        prompt = ''
        if self.style == 'json':
            prompt = json.dumps(self.tree_dict, indent=4)
        elif self.style == 'xml' or self.style == 'xml_wo_text':
            prompt = dict2xml(self.tree_dict, wrap="information", indent="\t")
        self.prompt = prompt

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return self.prompt
