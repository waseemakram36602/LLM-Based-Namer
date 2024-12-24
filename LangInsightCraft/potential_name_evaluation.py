from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class PotentialNameEvaluation:
    def __init__(self):
        pass

    def get_sentence_embedding(self, text: str):
        """Generates embeddings using a SentenceTransformer."""
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        return model.encode(text, convert_to_tensor=True)

    def calculate_edit_distance(self, generated_name: str, actual_name: str) -> int:
        """Calculates edit distance between the generated and actual method names."""
        return sum(1 for a, b in zip(generated_name, actual_name) if a != b)

    def calculate_semantic_similarity(self, generated_name: str, actual_name: str) -> float:
        """Calculates semantic similarity between two method names."""
        return util.pytorch_cos_sim(self.get_sentence_embedding(generated_name),
                                    self.get_sentence_embedding(actual_name)).item()

    def evaluate_method_name_score(self, edit_distance: int, semantic_similarity: float) -> float:
        """Evaluates the score of a method name based on edit distance and semantic similarity."""
        return (1 / (1 + edit_distance)) * semantic_similarity

    def evaluate_method_names(self, generated_methods: List[str], actual_method: str) -> List[Tuple[str, int, float]]:
        """Evaluates the generated method names based on edit distance and semantic similarity."""
        evaluated_scores = []
        for generated_method in generated_methods:
            edit_distance = self.calculate_edit_distance(generated_method, actual_method)
            semantic_similarity = self.calculate_semantic_similarity(generated_method, actual_method)
            score = self.evaluate_method_name_score(edit_distance, semantic_similarity)
            evaluated_scores.append((generated_method, edit_distance, semantic_similarity, score))
        
        evaluated_scores.sort(key=lambda x: x[3], reverse=True)
        return evaluated_scores
