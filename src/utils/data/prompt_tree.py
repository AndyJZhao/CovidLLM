import dgl
from omegaconf import DictConfig
import pandas as pd
import json
from copy import deepcopy
from utils.pkg.dict2xml import dict2xml
import re
from collections import OrderedDict


class PromptTree:
    def __init__(self, cfg, data, id, name_alias, style='xml', label=None):
        prompt = ''
        info_dict = {}
        if cfg.use_static_text:  # Add static text as prompt prefix
            prompt += data[id].Static_description + '\n'
        else:  # Add static text to info_dict to be further formatted
            info_dict['Static'] = data[id][cfg.data.static_cols].T.squeeze().to_dict()

        if cfg.use_dynamic_text:  # Add static text as prompt prefix
            prompt += data[id].Dynamic_description + '\n'

        for cont_field in cfg.data.dynamic_cols:
            if cfg.use_seq_encoder and cont_field in cfg.in_cont_fields:
                info_dict[cont_field] = f'<{cont_field.upper()}-EMB>'
            else:
                info_dict[cont_field] = str(data[id][cont_field])

        self.style = style
        self.label = label

        if self.style == 'json':
            prompt += json.dumps(info_dict, indent=4)
        elif self.style == 'xml':
            prompt += dict2xml(info_dict, wrap="information", indent="\t")
        self.prompt = prompt

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return self.prompt
