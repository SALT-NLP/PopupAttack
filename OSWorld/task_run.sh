#!/bin/bash

python run.py --path_to_vm /Users/zhangyanzhe/Documents/GitHub/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx --headless --observation_type som        --model gpt-4-turbo-2024-04-09 --test_all_meta_path evaluation_examples/test_easy.json --result_dir results_1 --temperature 0.0 --attack ./attack_config/intent_click_tgt_OK.json
sleep 30
python run.py --path_to_vm /Users/zhangyanzhe/Documents/GitHub/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx --headless --observation_type som        --model gpt-4-turbo-2024-04-09 --test_all_meta_path evaluation_examples/test_easy.json --result_dir results_2 --temperature 0.0
sleep 30
python run.py --path_to_vm /Users/zhangyanzhe/Documents/GitHub/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx --headless --observation_type screenshot --model gpt-4-turbo-2024-04-09 --test_all_meta_path evaluation_examples/test_easy.json --result_dir results_3 --temperature 0.0 --attack ./attack_config/intent_click_tgt_OK.json
sleep 30
python run.py --path_to_vm /Users/zhangyanzhe/Documents/GitHub/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx --headless --observation_type screenshot --model gpt-4-turbo-2024-04-09 --test_all_meta_path evaluation_examples/test_easy.json --result_dir results_4 --temperature 0.0