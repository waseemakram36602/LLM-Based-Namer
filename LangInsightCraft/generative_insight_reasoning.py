from typing import List

class GenerativeInsightReasoning:
    def __init__(self):
        pass

    def generate_reasoning_for_method(self, method_name: str, entities: List[str], actions: List[str], context_scope: str, edit_distance: int, semantic_similarity: float) -> str:
        """Generates reasoning for why a specific method name is the best choice."""
        return f"{method_name}, Reason Insight: Conveys {' and '.join(actions)} on {' and '.join(entities)} in the context of {context_scope}."

    def generate_all_insights(self, ranked_method_names: List[tuple], entities: List[str], actions: List[str], context_scope: str) -> List[str]:
        """Generates insights for all ranked method names."""
        insights = []
        for method_name, _, _, _ in ranked_method_names:
            insights.append(self.generate_reasoning_for_method(method_name, entities, actions, context_scope, 0, 0))
        return insights
