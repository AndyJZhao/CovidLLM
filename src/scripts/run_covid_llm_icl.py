import os
import sys

root_path = os.path.abspath(os.path.dirname(__file__)).split("src")[0]
os.chdir(root_path)
sys.path.append(root_path + "src")

from utils.basics import init_env_variables, time_logger, wandb_finish
from tqdm import tqdm

init_env_variables()

from utils.project.exp import init_experiment
import logging
import hydra

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from graph_text.icl import LLMForInContextLearning
from utils.data.textual_graph import TextualGraph
import torch as th
from llm import CpuFakeDebugLLM
from covid_llm.instruction_dataset import InstructionDataset
from torch.utils.data import Subset


@time_logger()
@hydra.main(config_path=f"{root_path}/configs", config_name="main_cfg", version_base=None)
def run_graph_text_inference(cfg):
    cfg, logger = init_experiment(cfg)
    data = TextualGraph(cfg=cfg)
    full_dataset = InstructionDataset(data, cfg, cfg.mode)
    dataset = Subset(full_dataset, data.split_ids.test[:cfg.data.max_test_samples])
    if cfg.get("debug", False):
        llm = CpuFakeDebugLLM()  # Use local CPU for faster debugging
    else:
        llm = hydra.utils.instantiate(cfg.llm)

    model = LLMForInContextLearning(cfg, data, llm, logger, **cfg.model)
    for i, item in tqdm(enumerate(dataset), "Evaluating..."):
        # for i, node_id in track(enumerate(data.split_ids.test[:10]), 'Evaluating...'):
        node_id, graph_tree_list, in_text, out_text, demo, question, _ = item
        is_evaluate = i % cfg.eval_freq == 0 and i != 0
        model(node_id, in_text, demo, question, log_sample=is_evaluate)
        if is_evaluate:
            model.eval_and_save(i, node_id)

    result = model.eval_and_save(i, node_id, final_eval=True)
    logger.info("Training finished")
    wandb_finish(result)


if __name__ == "__main__":
    run_graph_text_inference()
