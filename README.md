## Introduction

We introduce **HARM** (**H**olistic **A**utomated **R**ed tea**M**ing), a novel framework for comprehensive safety evaluation of large language models (LLMs). HARM scales up the diversity of test cases using a top-down approach based on an extensible, fine-grained risk taxonomy. Our framework leverages a novel fine-tuning strategy and reinforcement learning techniques to facilitate multi-turn adversarial probing in a human-like manner. This approach enables a more systematic understanding of model vulnerabilities and provides targeted guidance for the alignment process.

## Method

<img src=".\figs\method.png" style="zoom:63%;" />

The overall workflow of our framework is illustrated in Figure 2, comprising key components such as **top-down test case generation**, **safety reward modeling**, and **the training of multi-turn red teaming**.

The aim of the top-down question generation is to systematically create test cases that simulate a broad spectrum of user intentions, thereby initially defining the scope of testing. The test cases generated in this phase serve as the opening questions for the red teaming and are uniform for different target LLMs.

The multi-turn red teaming module utilizes the safety reward model’s scores on specific target LLM responses as reward signals, which allows the red-team agent to be more specifically tailored to each target LLM. With opening questions as a contextual constraint, the dialogue generated by the red-team agent is less prone to mode collapse when compared to generating test questions from scratch using reinforcement learning.

## Content of this repository

### Top-down Test Case Generation

- **async_top_down_generation.py**: Code for generating test questions using the direct method.
- **async_top_down_generation_attack_vectors.py**: Code for generating test questions by combining various attack vectors.
- **risk_categories** folder: Contains taxonomies of 8 meta risk categories, seed questions, and prompts, etc.

#### Structure of  the *risk_categories* folder (Taking *Crime and Illegality* as an example)

- **crime_attempts.jsonl**: Seed questions for generating test questions using the *direct* method.
- **prompt.txt**: Prompt for generating test questions using the *direct* method.
- **generated_questions.json**: Test questions generated using the *direct* method.
- **attack_vectors folder**:
  - xx.txt: Prompt for xx attack vector.
  - xx.jsonl: Seed questions for xx attack vector.
- **XX_questions.json**: Test questions generated using the XX attack vector.



### Multi-turn Red  Teaming

The corresponding implementation for each module:

- **Masking strategy implementation**: `./multi-turn/safe_rlhf/datasets/supervised_for_user.py`, `./multi-turn/safe_rlhf/finetune/trainer.py`

- **SFT for red-team agent**: `./multi-turn/scripts/red_teaming_sft_chat.sh`

- **Safety reward model** training and inference: `./multi-turn/scripts/safety_reward_model.sh`, `./multi-turn/multi-turn_reward_computation.py`

- **Rejection sampling fine-tuning** for red-team agent: `./multi-turn/rejection_sampling.py`, `./multi-turn/scripts/red_teaming_rs_chat.sh`

- **Multi-turn red teaming inference**: `./multi-turn/multi-turn_interaction.py`



## Data we use

- [Anthropic red-team-attempts](https://github.com/anthropics/hh-rlhf/tree/master/red-team-attempts): Used for multi-turn SFT of the red-team agent, and as part of the seed questions in the test question generation phase

- [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF): Part of the training set for the safety reward model

- [Anthropic harmless-base](https://github.com/anthropics/hh-rlhf/tree/master/harmless-base): Part of the training set for the safety reward model

- [HolisiticBias](https://github.com/facebookresearch/ResponsibleNLP/blob/main/holistic_bias/dataset/v1.0/descriptors.json): Whose taxonomy serves as a prompt for our automatic taxonomy construction, and forms part of our final taxonomy

## Acknowledgment

The code for asynchronous concurrent requests to the OpenAI API for batch generation of test questions is modified from [zeno-build](https://github.com/zeno-ml/zeno-build). Additionally, we used [PKU-SafeRLHF](https://github.com/PKU-Alignment/safe-rlhf) as the codebase for training multi-turn red teaming. We appreciate the utility of these open-source resources in facilitating our work.

## Citation

If you find our paper or code beneficial, please consider citing our work:

```
@inproceedings{
anonymous2024holistic,
title={Holistic Automated Red Teaming for Large Language Models through Top-Down Test Case Generation and Multi-turn Interaction},
author={Anonymous},
booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
year={2024},
url={https://openreview.net/forum?id=D0aoW6Re9j}
}
```



