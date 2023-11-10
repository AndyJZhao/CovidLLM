# Covid LLM

# Steps to reproduce

## Python Environment

```shell
pip install -r requirements.txt
```
### Supervised  Fine-tuning (SFT)
CovidLLM supports instruction fine-tuning a LLM on graph. An RNN is used to map the continuous sequence to text space (as tokens). We recommend to use BF16 for stable training.
```shell
cd src/scripts
python run_covid_llm_sft.py use_deepspeed=true use_wandb=true lora.r=-1 llm.base_model=llama2-7b total_steps=1000 cont_fields=all dropout=0.2 in_weeks=3 lr=2e-05 target=t1
```
## Key Hyper-parameters

- `encoder_type`: RNN, GRU, LSTM have similar performances; we use GRU as default.
- `total_steps`: Number of training steps.
- `use_static_text`: Static state description. 
- `use_dynamic_text`: Dynamic state description, includes vaccine, previous hospitalization of 12 weeks, and policy information (e.g. restriction on gathering).

