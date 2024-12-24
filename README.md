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
## Source Code
The source code complete project is in the folder [LangInsightCraft](https://github.com/waseemakram36602/LLM-Based-Namer/tree/main/LangInsightCraft)
## Class Hierarchy
![Hierarchy](ClassHirarchy.PNG)
### **Explanation of Key Classes and Methods:**

1. **BestExample**:
   - **Purpose**: This class is responsible for selecting the best functional descriptions from a CSV file based on semantic similarity to the input description.
   - **Key Methods**:
     - `find_top_n_similar_descriptions`: This method retrieves the top N similar descriptions from the CSV based on cosine similarity.

2. **ContextualInformationExtraction**:
   - **Purpose**: This class extracts contextual information (entities, actions, and context scope) from functional descriptions.
   - **Key Methods**:
     - `extract_entities`: Extracts entities (typically nouns).
     - `extract_actions`: Extracts actions (typically verbs).
     - `extract_context_scope`: Generates a context scope from the extracted entities and actions.

3. **PotentialNameEvaluation**:
   - **Purpose**: This class evaluates method names by calculating their edit distance and semantic similarity against the actual method name.
   - **Key Methods**:
     - `evaluate_method_names`: This method evaluates and ranks generated method names by computing their similarity to the actual method name.

4. **GenerativeInsightReasoning**:
   - **Purpose**: This class generates insights and reasoning behind each potential method name based on entities, actions, and context.
   - **Key Methods**:
     - `generate_reasoning_for_method`: Provides reasoning for a single method name.
     - `generate_all_insights`: Provides reasoning for all ranked method names.

5. **LangInsightCraft**:
   - **Purpose**: This class orchestrates all the steps involved in the process of generating method names. It combines the functionalities of all other classes to produce a context-enriched prompt for method name generation.
   - **Key Methods**:
     - `create_context_enriched_prompt`: This method creates the final context-enriched prompt for passing to the LLM for generating method names.

### **How It Works:**
1. **BestExample** class finds the most relevant examples from a CSV file based on semantic similarity to the input functional description.
2. **ContextualInformationExtraction** extracts the entities, actions, and context scope from these examples.
3. **PotentialNameEvaluation** evaluates and ranks the potential method names based on their edit distance and semantic similarity to the actual method name.
4. **GenerativeInsightReasoning** provides reasoning behind each ranked method name.
5. **LangInsightCraft** creates a context-enriched prompt using the outputs from all the classes and prepares it for use with a language model to generate method names.

---

The **LangInsightCraft** class integrates the entire process, providing an easy-to-use interface for generating method names from functional descriptions.
## Requirements

To run the LangInsightCraft project, you need to have the following dependencies installed. These dependencies can be easily installed using **pip**.

### System Requirements:
- Python 3.7 or higher

### Python Libraries:
- **torch**: Required for PyTorch-based embeddings and models.
- **sentence-transformers**: Used for generating sentence embeddings with pre-trained models like `all-mpnet-base-v2`.
- **pandas**: Used for reading and handling CSV files.
- **openai**: Required to interact with the OpenAI API for generating method names.

### Installation:

You can install all the required dependencies using the following command:

```bash
pip install torch sentence-transformers pandas openai

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
