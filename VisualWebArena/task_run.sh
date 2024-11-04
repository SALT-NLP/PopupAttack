#!/bin/bash

export HF_HOME="/data/yzhang3792"
export DATASET="visualwebarena"
export CLASSIFIEDS="http://127.0.0.1:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://127.0.0.1:7770"
export REDDIT="http://127.0.0.1:9999"
export WIKIPEDIA="http://127.0.0.1:8888"
export HOMEPAGE="http://127.0.0.1:4399"
export SHOPPING_ADMIN="http://127.0.0.1:7780/admin"
export GITLAB="http://127.0.0.1:8023"
export MAP="http://127.0.0.1:3000"
export OPENAI_API_KEY="sk-proj-XXX"
export ANTHROPIC_API_KEY="sk-ant-api03-XXX"

exp_id=1

curl -X POST http://127.0.0.1:9980/index.php?page=reset -d "token=4b61655535e7ed388f0d40a93600254c"
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
mkdir -p ./.auth
python browser_env/auto_login.py
python run.py --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx 0 --test_end_idx 72 --result_dir result${exp_id} --test_config_base_dir=agent_easy --model gpt-4-turbo-2024-04-09 --action_set_tag som --observation_type image_som --max_obs_length 10000 --eval_captioning_model_device cuda --temperature 0.0 > output_${exp_id}_run3.txt 2>&1

exp_id=2

curl -X POST http://127.0.0.1:9980/index.php?page=reset -d "token=4b61655535e7ed388f0d40a93600254c"
bash scripts/reset_reddit.sh
bash scripts/reset_shopping.sh
mkdir -p ./.auth
python browser_env/auto_login.py
python run.py --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx 0 --test_end_idx 72 --result_dir result${exp_id} --test_config_base_dir=agent_easy --model gpt-4-turbo-2024-04-09 --action_set_tag som --observation_type image_som --attack ./attack_config/intent_click_tag_OK_adv_text.json --max_obs_length 10000 --eval_captioning_model_device cuda --temperature 0.0 > output_${exp_id}_run3.txt 2>&1