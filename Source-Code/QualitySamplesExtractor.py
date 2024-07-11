import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize
import torch

class QualitySamplesExtractor:
    def __init__(self, training_dataset, bert_model_name='bert-base-uncased'):
        """
        Initialize the QualitySamplesExtractor with a training dataset and a pre-trained BERT model.

        :param training_dataset: List of tuples [(functional_description, method_name), ...]
        :param bert_model_name: Name of the pre-trained BERT model to use.
        """
        self.training_dataset = training_dataset
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)
        self.high_quality_samples = []
        self.alpha = None
        self.beta = None

    def encode_text(self, text):
        """
        Encode text into a vector using BERT.

        :param text: The text to encode.
        :return: The vector representation of the text.
        """
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def semantic_similarity(self, vec_d, vec_m):
        """
        Calculate the semantic similarity between two vectors using cosine similarity.

        :param vec_d: Vector representation of the functional description.
        :param vec_m: Vector representation of the method name.
        :return: The cosine similarity score between the two vectors.
        """
        return cosine_similarity(vec_d, vec_m)[0][0]

    def length_alignment(self, d, m):
        """
        Calculate the length alignment score between a functional description and a method name.

        :param d: The functional description.
        :param m: The method name.
        :return: The length alignment score.
        """
        len_d = len(d)
        len_m = len(m)
        return 1 - abs(len_d - len_m) / max(len_d, len_m)

    def weighted_score(self, d, m):
        """
        Calculate the weighted score for a functional description and a method name.

        :param d: The functional description.
        :param m: The method name.
        :return: The weighted score combining semantic similarity and length alignment.
        """
        vec_d = self.encode_text(d)
        vec_m = self.encode_text(m)
        ss = self.semantic_similarity(vec_d, vec_m)
        la = self.length_alignment(d, m)
        return self.alpha * ss + self.beta * la

    def objective_function(self, params, validation_dataset):
        """
        Objective function for hyperparameter tuning.

        :param params: The parameters (alpha and beta) to optimize.
        :param validation_dataset: Validation dataset for calculating the error.
        :return: The mean squared error between the weighted scores and the ground truth similarity scores.
        """
        self.alpha, self.beta = params
        errors = []
        for d, m, ground_truth in validation_dataset:
            ws = self.weighted_score(d, m)
            errors.append((ws - ground_truth) ** 2)
        return np.mean(errors)

    def tune_hyperparameters(self, validation_dataset):
        """
        Tune the hyperparameters alpha and beta using the validation dataset.

        :param validation_dataset: Validation dataset for tuning.
        """
        initial_params = [0.5, 0.5]
        bounds = [(0.1, 0.9), (0.1, 0.9)]
        result = minimize(self.objective_function, initial_params, args=(validation_dataset,), bounds=bounds)
        self.alpha, self.beta = result.x

    def select_high_quality_samples(self, top_n=2000):
        """
        Select the top N high-quality samples based on the weighted score.

        :param top_n: The number of top samples to select.
        :return: List of top N high-quality samples.
        """
        if self.alpha is None or self.beta is None:
            raise ValueError("Hyperparameters alpha and beta must be tuned before selecting high-quality samples.")
        
        scores = []
        for d, m in self.training_dataset:
            ws = self.weighted_score(d, m)
            scores.append((d, m, ws))
        scores.sort(key=lambda x: x[2], reverse=True)
        self.high_quality_samples = scores[:top_n]
        return self.high_quality_samples

    def divide_samples(self):
        """
        Divide the high-quality samples into four distinct datasets.

        :return: Four datasets: TuningDS, InputDS, CandidateDS, and EvaluationDS.
        """
        num_samples = len(self.high_quality_samples)
        chunk_size = num_samples // 4
        tuning_ds = self.high_quality_samples[:chunk_size]
        input_ds = self.high_quality_samples[chunk_size:2*chunk_size]
        candidate_ds = self.high_quality_samples[2*chunk_size:3*chunk_size]
        evaluation_ds = self.high_quality_samples[3*chunk_size:]
        return tuning_ds, input_ds, candidate_ds, evaluation_ds
