# Replication package for the paper: "A Cross-Language Approach to Recommending Method Names According to Functional Descriptions"

# Introduction:
This paper introduces a novel approach to automatically suggesting high-quality Java method names using large language models (LLMs). Leveraging LLMs' advanced understanding capabilities for natural language descriptions of method functionalities, this approach introduces an algorithm called LangInsightCraft to generate context-enriched prompts that align generated names with established naming conventions, enhancing code readability and maintainability.
# Datasets:
There are two datasets are used to evalatute the approach
- [English Dataset:](https://github.com/waseemakram36602/LLM-Based-Namer/tree/main/Datasets/English-Dataset) Method Names with English Functional Descriptions (Dataset of Baseline).
- [Chinese Dataset:](https://github.com/waseemakram36602/LLM-Based-Namer/tree/main/Datasets/Chinese-Dataset) Method Names with Chinese Functional Descriptions. The Dataset organized from [Java 11 API Reference](https://www.apiref.com/java11-zh/java.base/module-summary.html)
- [Unseen Dataset:](https://github.com/waseemakram36602/LLM-Based-Namer/tree/main/Datasets/UnseenData) Method Names with Chinese Functional Descriptions from offline private dataset. 
Each dataset is crucial for training and evaluating the models to ensure they perform effectively across linguistic boundaries.
# LangInsightCraft
# BaseLines
The source code for the applied baseline approaches is provided below: 
## Deep Learning-based Baseline
- [RNN-att-Copy](https://github.com/waseemakram36602/LLM-Based-Namer/blob/main/Baseline_Code/RNN_Copy_Attn/RNN_README.md) 
- [CodeBERT](https://github.com/waseemakram36602/LLM-Based-Namer/blob/main/Baseline_Code/CodeBERT/CodeBERT_README.md)
- [UniXcoder](https://github.com/waseemakram36602/LLM-Based-Namer/blob/main/Baseline_Code/UniXcoder/UniXcoder_README.md)
## Large Language Model-based Baselines
- [ChatGPT-4o](https://github.com/waseemakram36602/LLM-Based-Namer/blob/main/LLMs-Code/ChatGPT/ChatGPT_README.md)
- [Llama 3](https://github.com/waseemakram36602/LLM-Based-Namer/blob/main/LLMs-Code/Llama3/Llama3_README.md) 
- [Gemini 1.5](https://github.com/waseemakram36602/LLM-Based-Namer/blob/main/LLMs-Code/Gemini/Gemini_README.md)


# LangInsightCraft
LangInsightCraft | |-- BestExample | |-- init(self) | |-- get_sentence_embedding(self, text) | |-- find_top_n_similar_descriptions(self, input_description, n=10) | |-- ContextualInformationExtraction (CIE) | |-- init(self) | |-- extract_entities(self, description) | |-- extract_actions(self, description) | |-- extract_context_scope(self, description, entities, actions) | |-- extract_contextual_info(self, description) | |-- PotentialNameEvaluation (PNE) | |-- init(self) | |-- calculate_edit_distance(self, generated_name, actual_name) | |-- calculate_semantic_similarity(self, generated_name, actual_name) | |-- evaluate_method_name_score(self, edit_distance, semantic_similarity) | |-- evaluate_method_names(self, generated_methods, actual_method) | |-- GenerativeInsightReasoning | |-- init(self) | |-- generate_reasoning_for_method(self, method_name, entities, actions, context_scope, edit_distance, semantic_similarity) | |-- generate_all_insights(self, ranked_method_names, entities, actions, context_scope) | |-- Main | |-- init(self) | |-- run(self, input_description, csv_file_path)
## Example usage
if __name__ == "__main__":
    LangInsightCraft = LangInsightCraft(train_file='java_train.csv', test_file='java_test.csv', valid_file='java_valid.csv')
    LangInsightCraft.run()
    
This snippet gives a clear, step-by-step guide for users to replicate the study, ensuring they understand how to set up their environment correctly. Make sure to include any additional specific instructions or prerequisites needed directly in your README or linked documentation to assist users further.
git clone https://github.com/waseemakram36602/LLM-Based-Namer.git

Getting Started
To get started with LangInsightCraft:
Install the required dependencies: pip install -r requirements.txt
Follow the instructions in the usage_instructions.md file for detailed steps on how to train and evaluate the models using the provided datasets and prompts.
Contribution
Contributions to LLM-MethodNamer are welcome. If you have suggestions for improvement or want to report bugs, please open an issue or submit a pull request.
