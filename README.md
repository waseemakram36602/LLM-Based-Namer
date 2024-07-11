![Banner](Mnamer.png)
# Replication package for paper : "Automated Suggestion of Method Names According to Functional Descriptions"

# Introduction:
mNamer introduces a novel approach to automatically suggesting high-quality Java method names using large language models (LLMs). Leveraging LLMs' advanced understanding capabilities for natural language descriptions of method functionalities, mNamer combines specialized pre-processing and customized post-processing techniques. This method aims to align generated names with established naming conventions, enhancing code readability and maintainability.
# Datasets:
There are two datasets are used to evalatute the approach
- [English Dataset:](https://github.com/propaki/Automethod/tree/main/EnglishDataset) Method Names with English Functional Descriptions (Dataset of Baseline).
- [Chinese Dataset:](https://github.com/propaki/Automethod/tree/main/Chinese%20Dataset) Method Names with Chinese Functional Descriptions. The Dataset organized from [Java 11 API Reference](https://www.apiref.com/java11-zh/java.base/module-summary.html)
Each dataset is crucial for training and evaluating the models to ensure they perform effectively across linguistic boundaries.
#  [Optimal Prompts:](https://github.com/propaki/Automethod/tree/main/OptiPrompts) 
Included in the "[OptiPrompts](https://github.com/propaki/Automethod/tree/main/OptiPrompts)" folder is a carefully curated corpus of prompts, comprised of text files, designed to enhance the performance of ChatGPT in accurately generating Java method names based on functional descriptions. These prompts are crafted to elicit precise and contextually relevant responses from the model, adhering to a well-designed template that aligns with the naming conventions and requirements specific to Java methods.
![Prompt Corpus](Optiprompts.PNG)
#Source-Code
Source code of our approach available in Source code folder
#Class Hierarchy
mNamer

- mNamer Class: Main class to handle the entire process, including loading datasets, extracting quality samples, generating prompt templates, fine-tuning models, generating in-context prompts, and ranking responses.
![mNamer.py](mNamer.PNG)

- QualitySamplesExtractor Class: Handles extracting high-quality samples, calculating similarity scores, and tuning hyperparameters.
![QualitySamplesExtractor.py](QualitySample.PNG)
- SFTCorpusFormation Class: Forms the SFT corpus by creating JSON entries and saving them to a file.
![SFTCorpusFormation.py](SFTCorpus.PNG)
- BestExampleSelection Class: Selects the best examples from the candidate dataset based on text similarity.
![BestExampleSelection.py](BestExamples.PNG)
- InContextPromptGenerator Class: Generates in-context prompts by integrating functional descriptions and best examples into the prompt template.
![InContextPromptGenerator.py](InContextPrompts.PNG)
- GPTFineTuner Class: Handles fine-tuning the GPT model using OpenAI's API.
![GPTFineTuner.py](GPTFine.PNG)
- GeminiFineTuner Class: Handles fine-tuning the Gemini model using Google Cloud's Vertex AI.
![GeminiFineTuner.py](GeminiFine.PNG)
- LlamaFineTuner Class: Handles fine-tuning the Llama model using the Hugging Face Transformers library.
![LlamaFineTuner.py](LlamaFine.PNG)
- RewardModel Class: Calculates semantic similarity, edit distance, and reward scores for method names.
![RewardModel.py](RewardModel.PNG)
- ALARO Class: Dynamically adjusts weights for reward optimization using a neural network.
![ALARO.py](ALARO.PNG)
- ResponseRanker Class: Ranks suggested method names based on reward scores and generates feedback prompts.
![ResponseRanker.py](Response.PNG)

# [RNN-Attn-Copy (Baseline Model):](https://github.com/propaki/Automethod/tree/main/Source-Code/RNN-Attn-Copy.ipynb)
We meticulously reproduced and implemented the baseline model in "[Source-Code](https://github.com/propaki/Automethod/tree/main/Source-Code)", which is a [RNN-Attn-Copy](https://github.com/propaki/Automethod/tree/main/Source-Code/RNN-Attn-Copy.ipynb) equipped with both attention and copying mechanisms. This advanced architecture was chosen as our benchmark for assessing the performance of alternative models due to its proven prowess in sequence-to-sequence translation tasks and its exceptional ability to grasp contextual nuances within sequences.

# mNamer

This snippet gives a clear, step-by-step guide for users to replicate the study, ensuring they understand how to set up their environment correctly. Make sure to include any additional specific instructions or prerequisites needed directly in your README or linked documentation to assist users further.
git clone https://github.com/propaki/Automethod.git

fine-tuned ready to chat ChatGPT extention availabe at [Mnamer](https://chat.openai.com/g/g-T58v7ELEM-mnamer)
The source code is the centerpiece of this repository, showcasing the application of BERT-based semantic model for both Semantic Driven Preprocessing and BERT-based RLHF in Postprocessing for LLMs to improve its method naming capabilities. This model represents a significant advancement in the field of automated method naming.

Getting Started
To get started with mNamer:
Install the required dependencies: pip install -r requirements.txt
Follow the instructions in the usage_instructions.md file for detailed steps on how to train and evaluate the models using the provided datasets and prompts.
Contribution
Contributions to LLM-MethodNamer are welcome. If you have suggestions for improvement or want to report bugs, please open an issue or submit a pull request.
