# Attacking Vision-Language Computer Agents via Pop-ups

*Yanzhe Zhang, Tao Yu, Diyi Yang*

## Overview

Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing the tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.

![](./assets/intro.svg)

## Set Up

Please follow the setup instructions of OSWorld and VisualWebArena. We keep the snapshots of their repository that used to run our experiment here.

## Data

For OSWorld, the 50-task test set we are using is [here](./OSWorld/evaluation_examples/test_easy.json). For 72-task test set we are using is [here](./VisualWebArena/agent_easy/).

## Code Structure

We outline the files that we made significant changes.

```
|_OSWorld/
      |_attack.py -> main logic of drawing adversaril pop-ups
      |_general_attack_utils.py -> drawing utils
      |_image_processing.py -> drawing utils
      |_mm_agents/
            |_agent.py -> agent prediction, preparation of attack
|_VisualWebArena/
      |_browswer_env/
            |_processors.py -> agent prediction, preparation of attack
      |_run.py
```

## Example Command

```bash
cd OSWorld
bash ./task_run.sh
```

```bash
cd VisualWebArena
bash ./task_run.sh
```


## Acknowledgement

We thank [OSWorld](https://github.com/xlang-ai/OSWorld), [VisualWebArena](https://github.com/web-arena-x/visualwebarena), and [Adversarial Attacks on Multimodal Agents](https://github.com/ChenWu98/agent-attack) for their awsome work!

If you find our work helpful, please consider citing our paper:

```
@misc{zhang2024attacking,
      title={Attacking Vision-Language Computer Agents via Pop-ups}, 
      author={Yanzhe Zhang and Tao Yu and Diyi Yang},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

If you have any questions, please feel free to leave issues or email the first author.
