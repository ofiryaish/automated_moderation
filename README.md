# Leveraging Language Models for Automated Moderation


The code we developed in coursework where we gained experience with large language models and fine-tuned the state-of-the-art model Mistral-7B-Instruct for automated moderation tasks in the r/ChangeMyView forum (CMV).

# Usage
1. `utilities.py` contains most of the functions for generating the prompt and the moderation branch context.
2. `chat_gpt_generate_moderations.ipynb` contains the code for generating the "ideal" moderations using chatGPT API for fine-tuning.
3. `fine_tune.ipynb` contains the code for fine-tuning the Mistral-7B-Instruct model with the synthetic moderations replying to negative toned utterances.
4. `fine_tune_with_positives.ipynb` contains the code for fine-tuning the Mistral-7B-Instruct model with positive-like utterances and synthetic moderations replying to negative toned utterances.
5. `chat_gpt_generate_ratings.ipynb` contains the code for the automatic rating using ChatGPT API.

# Datasets included
1. `annotated_trees_101.csv` - The tagged CMV data from [Discourse Parsing for Contentious, Non-Convergent Online Discussions](https://ojs.aaai.org/index.php/ICWSM/article/view/18109).
2. `bad_tone_nodes_with_generated_messages_chat_gpt_3_5_turbo.csv`  - The synthetic moderations replying to negative toned utterances generated using ChatGPT 3.5-Turbo API.

Our coursework report is included.
