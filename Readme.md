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
## Dataset 
Hospitalization data:
- Risk_level: Low (hospitalization_per_100k < 10), Median (10 < hospitalization_per_100k < 20), High (hospitalization_per_100k > 20).
- Trend: Percent change in the current weekly total new COVID-19 hospitalization per 100k population compared with the prior week. Substantial Decrease (% Change from prior week < -0.2), Moderate Decrease (-0.2 < % Change from prior week < -0.1), stable (-0.1 < % Change from prior week < 0.1), Moderate Decrease (0.1 < % Change from prior week < 0.2), Substantial Decrease (0.2 < % Change from prior week)
## Key Hyper-parameters

- `target`: The target of classification, valid choices: r1, r2, r3, r4, t1, t2, t3, t4
  - Risk level
    - `r1`: True `Risk_level` one week later.
    - `r2`: True `Risk_level` two weeks later.
    - `r3`: True `Risk_level` three weeks later.
    - `r4`: True `Risk_level` four weeks later.
  - Trend
    - `t1`: True `Trend` one week later.
    - `t2`: True `Trend` two weeks later.
    - `t3`: True `Trend` three weeks later.
    - `t4`: True `Trend` four weeks later.
- `encoder_type`: RNN, GRU, LSTM have similar performances; we use GRU as default.
- `total_steps`: Number of training steps.
- `use_static_text`: Static state description. 
- `use_dynamic_text`: Dynamic state description, includes vaccine, previous hospitalization of 12 weeks, and policy information (e.g. restriction on gathering).
- `cont_fields`: Fields of continuous sequential data.
- `llm.base_model`: `llama2-7b` as default, requires 48 GB GPU memory. Use `llm.base_model=tinygpt` for debugging.

