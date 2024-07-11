import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize
import json
import Levenshtein
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import datasets
import openai
from google.cloud import aiplatform
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class mNamer:
    def __init__(self, train_file, test_file, valid_file, bert_model_name='bert-base-uncased'):
        self.train_file = train_file
        self.test_file = test_file
        self.valid_file = valid_file
        self.bert_model_name = bert_model_name

        self.train_dataset = self.load_dataset(train_file)
        self.test_dataset = self.load_dataset(test_file)
        self.valid_dataset = self.load_dataset(valid_file)
        
        self.qs_extractor = QualitySamplesExtractor(self.train_dataset, bert_model_name)
        self.sft_corpus_formation = None
        self.best_example_selector = None
        self.in_context_prompt_generator = None
        self.gpt_fine_tuner = None
        self.gemini_fine_tuner = None
        self.llama_fine_tuner = None
        self.reward_model = RewardModel()
        self.alaro = ALARO()
        self.response_ranker = ResponseRanker(self.reward_model, self.alaro)

    def load_dataset(self, file_path):
        df = pd.read_csv(file_path)
        dataset = list(zip(df['functional_description'], df['method_name']))
        return dataset

    def extract_quality_samples(self):
        self.qs_extractor.tune_hyperparameters(self.valid_dataset)
        high_quality_samples = self.qs_extractor.select_high_quality_samples()
        tuning_ds, input_ds, candidate_ds, evaluation_ds = self.qs_extractor.divide_samples()
        self.sft_corpus_formation = SFTCorpusFormation(tuning_ds)
        self.best_example_selector = BestExampleSelection(candidate_ds)
        return tuning_ds, input_ds, candidate_ds, evaluation_ds

    def generate_prompt_templates(self):
        prompt_templates = [
            """Query: Provide a method name based on the given functional description, considering the thought process and constraints
               Functional Description: {functional_description}
                Output Requirements:
                1. Identify the main action being performed (verb).
                2. Determine the main object being acted upon (noun).
                3. Ensure the method name is in Camel-Case format, starting with lowercase and capitalizing each following word.
                4. Make the method name concise and meaningful.
                5. Avoid Java-reserved words or keywords.
                6. Do not include special characters or spaces in the method name.

                Examples:
                    {examples_section}"""
               ]
        return prompt_templates

    def fine_tune_models(self, sft_corpus_file):
        # Fine-tune GPT
        self.gpt_fine_tuner = GPTFineTuner(api_key="YOUR_OPENAI_API_KEY")
        file_id = self.gpt_fine_tuner.upload_corpus(sft_corpus_file)
        job_id = self.gpt_fine_tuner.fine_tune(file_id)
        status = self.gpt_fine_tuner.monitor_fine_tuning(job_id)
        print(f"GPT fine-tuning completed with status: {status}")

        # Fine-tune Gemini
        self.gemini_fine_tuner = GeminiFineTuner(project_id="YOUR_PROJECT_ID", location="us-central1")
        corpus_uri = self.gemini_fine_tuner.upload_corpus(sft_corpus_file)
        model_name = self.gemini_fine_tuner.fine_tune(corpus_uri)
        print(f"Fine-tuned Gemini model name: {model_name}")

        # Fine-tune Llama
        self.llama_fine_tuner = LlamaFineTuner(model_name='facebook/llama-3', output_dir='./llama_results')
        dataset = self.llama_fine_tuner.load_dataset(sft_corpus_file)
        self.llama_fine_tuner.fine_tune(dataset)
        print(f"Llama fine-tuning completed. Model saved to: {self.llama_fine_tuner.output_dir}")

    def generate_in_context_prompt(self, input_description, optimal_prompt_template):
        best_examples = self.best_example_selector.select_best_examples(input_description)
        examples_section = ""
        for example in best_examples:
            example_description = example[0]
            example_method_name = example[1]
            examples_section += f"Functional Description: {example_description}\nMethod Name: {example_method_name}\n\n"

        in_context_prompt = optimal_prompt_template.replace("{functional_description}", input_description)
        in_context_prompt = in_context_prompt.replace("{examples_section}", examples_section.strip())
        return in_context_prompt

    def rank_responses_and_generate_feedback(self, input_description, suggested_method_names):
        ranked_methods = self.response_ranker.rank_responses(input_description, suggested_method_names)
        feedback_prompt = self.response_ranker.generate_feedback_prompt(input_description, ranked_methods)
        print(feedback_prompt)
        return feedback_prompt

    def run(self):
        # Step 1: Extract quality samples
        tuning_ds, input_ds, candidate_ds, evaluation_ds = self.extract_quality_samples()
        
        # Step 2: Create SFT corpus and save to file
        sft_corpus = self.sft_corpus_formation.create_corpus()
        sft_corpus_file = "SFT_corpus.jsonl"
        with open(sft_corpus_file, 'w', encoding='utf-8') as f:
            json.dump(sft_corpus, f, ensure_ascii=False, indent=4)
        
        # Step 3: Fine-tune models
        self.fine_tune_models(sft_corpus_file)
        
        # Step 4: Generate in-context prompt for a test example
        test_input_description = input_ds[0][0]  # Assuming input_ds has at least one item
        optimal_prompt_template = self.generate_prompt_templates()[0]  # Temporarily using the first template
        in_context_prompt = self.generate_in_context_prompt(test_input_description, optimal_prompt_template)
        print(f"In-context Prompt:\n{in_context_prompt}")
        
        # Step 5: Simulate model responses (example method names)
        suggested_method_names = ["deleteItemListener", "detachItemListener", "eraseItemListener", "removeItemHandler"]
        
        # Step 6: Rank responses and generate feedback
        feedback_prompt = self.rank_responses_and_generate_feedback(test_input_description, suggested_method_names)
        print(f"Feedback Prompt:\n{feedback_prompt}")

# Example usage
if __name__ == "__main__":
    mnamer = mNamer(train_file='java_train.csv', test_file='java_test.csv', valid_file='java_valid.csv')
    mnamer.run()
