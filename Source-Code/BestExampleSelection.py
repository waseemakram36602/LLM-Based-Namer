from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BestExampleSelection:
    def __init__(self, candidate_dataset, model_name='bert-base-uncased'):
        """
        Initialize the BestExampleSelection with a candidate dataset and a pre-trained BERT model.

        :param candidate_dataset: List of tuples [(functional_description, method_name), ...]
        :param model_name: Name of the pre-trained BERT model to use.
        """
        self.candidate_dataset = candidate_dataset
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode_text(self, text):
        """
        Encode text into a vector using BERT.

        :param text: The text to encode.
        :return: The vector representation of the text.
        """
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def text_similarity(self, fd, e_d):
        """
        Calculate the text similarity between two functional descriptions.

        :param fd: Functional description of the input.
        :param e_d: Functional description of the example.
        :return: The cosine similarity score between the two descriptions.
        """
        vec_fd = self.encode_text(fd)
        vec_e_d = self.encode_text(e_d)
        return cosine_similarity(vec_fd, vec_e_d)[0][0]

    def select_best_examples(self, fd, top_n=30):
        """
        Select the top N best examples from the candidate dataset based on similarity to the input functional description.

        :param fd: Functional description of the input.
        :param top_n: Number of top examples to select.
        :return: List of top N best examples [(functional_description, method_name, similarity_score), ...]
        """
        scores = []
        for e_d, e_m in self.candidate_dataset:
            similarity_score = self.text_similarity(fd, e_d)
            scores.append((e_d, e_m, similarity_score))

        scores.sort(key=lambda x: x[2], reverse=True)
        best_examples = scores[:top_n]
        return best_examples
