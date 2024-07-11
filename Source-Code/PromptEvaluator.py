from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

class PromptEvaluator:
    def __init__(self, evaluation_dataset, prompt_templates, model_name='gpt-3.5-turbo'):
        """
        Initialize the PromptEvaluator with an evaluation dataset, prompt templates, and an LLM model.

        :param evaluation_dataset: List of tuples [(functional_description, method_name), ...]
        :param prompt_templates: List of prompt templates.
        :param model_name: The name of the LLM model to use (e.g., 'gpt-3.5-turbo').
        """
        self.evaluation_dataset = evaluation_dataset
        self.prompt_templates = prompt_templates
        self.model = pipeline('text-generation', model=model_name)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def encode_text(self, text):
        """
        Encode text into a vector using BERT.

        :param text: The text to encode.
        :return: The vector representation of the text.
        """
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def semantic_similarity(self, vec_d, vec_m):
        """
        Calculate the semantic similarity between two vectors using cosine similarity.

        :param vec_d: Vector representation of the functional description.
        :param vec_m: Vector representation of the method name.
        :return: The cosine similarity score between the two vectors.
        """
        return cosine_similarity(vec_d, vec_m)[0][0]

    def evaluate_prompt(self, prompt_template, functional_description, ground_truth_method_name):
        """
        Evaluate a single prompt template.

        :param prompt_template: The prompt template to evaluate.
        :param functional_description: The functional description to insert into the prompt template.
        :param ground_truth_method_name: The ground truth method name for comparison.
        :return: The similarity score between the generated method name and the ground truth.
        """
        prompt = prompt_template.replace("{functional_description}", functional_description)
        generated_method_name = self.model(prompt)[0]['generated_text'].strip()
        vec_generated = self.encode_text(generated_method_name)
        vec_ground_truth = self.encode_text(ground_truth_method_name)
        return self.semantic_similarity(vec_generated, vec_ground_truth)

    def evaluate_all_prompts(self):
        """
        Evaluate all prompt templates on the evaluation dataset and select the optimal prompt template.

        :return: The optimal prompt template with the highest average similarity score.
        """
        optimal_prompt = None
        highest_avg_name_sim = -1

        for prompt_template in self.prompt_templates:
            total_name_sim = 0
            for functional_description, ground_truth_method_name in self.evaluation_dataset:
                name_sim = self.evaluate_prompt(prompt_template, functional_description, ground_truth_method_name)
                total_name_sim += name_sim
            avg_name_sim = total_name_sim / len(self.evaluation_dataset)

            if avg_name_sim > highest_avg_name_sim:
                highest_avg_name_sim = avg_name_sim
                optimal_prompt = prompt_template

        return optimal_prompt, highest_avg_name_sim
