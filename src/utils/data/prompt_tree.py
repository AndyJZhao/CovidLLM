import dgl
from omegaconf import DictConfig
import pandas as pd
import json
from copy import deepcopy
from utils.pkg.dict2xml import dict2xml
import re
from collections import OrderedDict

class PromptTree:
    def __init__(self, cfg, data, id, hierarchy, name_alias, style='xml', label=None):
        self.info_dict = info_dict = OrderedDict({
            'Static': data.df.iloc[id][cfg.data.static_cols].T.squeeze().to_dict(),
            'Dynamic': data.df.iloc[id][cfg.data.dynamic_cols].T.squeeze().to_dict(),
        })

        self.style = style
        self.hierarchy = hierarchy
        self.label = label
        prompt = ''
        if self.style == 'json':
            prompt = json.dumps(info_dict, indent=4)
        elif self.style == 'xml' or self.style == 'xml_wo_text':
            prompt = dict2xml(info_dict, wrap="information", indent="\t")
        self.prompt = prompt

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return self.prompt
