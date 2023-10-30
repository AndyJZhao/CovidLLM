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
        static_info = data[id].Static_description if cfg.use_static_text \
            else data[id][cfg.data.static_cols].T.squeeze().to_dict()
        info_dict = data[id][cfg.data.dynamic_cols].T.squeeze().to_dict()
        if cfg.use_seq_encoder:
            for cont_field in cfg.in_cont_fields:
                info_dict[cont_field] = f'<{cont_field.upper()}-EMB>'
        self.style = style
        self.hierarchy = hierarchy
        self.label = label
        prompt = static_info + '\n'
        if self.style == 'json':
            prompt += json.dumps(info_dict, indent=4)
        elif self.style == 'xml':
            prompt += dict2xml(info_dict, wrap="information", indent="\t")
        self.prompt = prompt

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return self.prompt
