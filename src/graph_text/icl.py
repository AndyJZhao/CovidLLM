from omegaconf import DictConfig

from llm.llm import LLM
from utils.data.covid_data import CovidData


class LLMForInContextLearning(object):
    def __init__(self, cfg: DictConfig, data: CovidData, llm: LLM, _logger, max_new_tokens=20, gen_mode="text",
                 **kwargs, ):
        self.cfg = cfg
        self.gen_mode = gen_mode
        self.data = data
        self.df = data.df
        self.logger = _logger
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        # ! Classification prompt

        self.df["dialog"] = "NA"
        self.df["demo"] = "NA"
        self.df["question"] = "NA"
        self.df["generated_text"] = "NA"

    def eval_and_save(self, step, node_id, final_eval=False):
        res_df = self.df.dropna()
        res_df["correctness"] = res_df.apply(lambda x: x["gold_choice"] in x["pred_choice"], axis=1)
        res_df.sort_values('correctness', inplace=True)
        res_df.to_csv(self.cfg.save_file)
        acc = res_df["correctness"].mean()
        self.logger.save_file_to_wandb(self.cfg.save_file, base_path=self.cfg.out_dir)
        valid_choice_rate = (res_df["pred_choice"].isin(self.data.choice_to_label_id.keys()).mean())
        acc_in_valid_choice = acc / valid_choice_rate if valid_choice_rate > 0 else 0
        result = {
            "acc": acc,
            "valid_choice_rate": valid_choice_rate,
            "acc_in_valid_choice": acc_in_valid_choice,
        }
        if valid_choice_rate > 0:
            valid_df = res_df[res_df["pred_choice"].isin(self.data.choice_to_label_id.keys())]
            valid_df["true_choices"] = valid_df.apply(lambda x: self.data.label_info.choice[x["label_id"]], axis=1)
            result.update({f"PD/{choice}.{self.data.choice_to_label_name[choice]}": cnt / len(valid_df)
                           for choice, cnt in valid_df.pred_choice.value_counts().to_dict().items()})
        sample = {f"sample_{k}": v
                  for k, v in self.data[node_id].to_dict().items()}
        self.logger.info(sample)
        self.logger.wandb_metric_log({**result, "step": step})

        #  ! Save statistics to results
        # y_true, y_pred = [valid_df.apply(lambda x: self.data.l_choice_to_id[x[col]], axis=1).tolist() for col in
        #                   ('true_choices', 'pred_choice')]
        # result['cla_report'] = classification_report(
        #     y_true, y_pred, output_dict=True,
        #     target_names=self.data.label_info.label_name.tolist())
        result["out_file"] = self.cfg.save_file
        self.logger.info(result)
        self.logger.critical(f"Saved results to {self.cfg.save_file}")
        self.logger.wandb_summary_update({**result, **sample}, finish_wandb=final_eval)
        return result

    def __call__(self, node_id, prompt, demo, question, log_sample=False):
        # ! Classification
        prompt = prompt + " " if prompt.endswith(":") else prompt  # ! Critical
        if self.gen_mode == "choice":
            generated = self.llm.generate_text(prompt, max_new_tokens=1, choice_only=True)
            pred_choice = generated[-1] if len(generated) > 0 else "NULL"
        else:
            generated = self.llm.generate_text(prompt, self.max_new_tokens)
            try:
                pred_choice = generated.split("<answer>")[-1][0]  # Could be improved
                print(pred_choice)
            except:
                pred_choice = ""

        self.df.loc[node_id, "dialog"] = prompt + generated
        self.df.loc[node_id, "demo"] = demo
        self.df.loc[node_id, "question"] = question
        self.df.loc[node_id, "pred_choice"] = pred_choice
        self.df.loc[node_id, "generated_text"] = generated
