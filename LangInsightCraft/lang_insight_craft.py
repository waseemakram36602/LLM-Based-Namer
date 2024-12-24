from best_example import BestExample
from context_information import ContextualInformationExtraction
from potential_name_evaluation import PotentialNameEvaluation
from generative_insight_reasoning import GenerativeInsightReasoning
import pandas as pd

class LangInsightCraft:
    def __init__(self, csv_file_path: str):
        self.best_example = BestExample(csv_file_path)
        self.cie = ContextualInformationExtraction()
        self.pne = PotentialNameEvaluation()
        self.gir = GenerativeInsightReasoning()

    def create_context_enriched_prompt(self, input_description: str, n: int = 10) -> str:
        """Generates a context-enriched prompt for method name generation."""
        best_examples = self.best_example.find_top_n_similar_descriptions(input_description, n)
        contextual_info = self.cie.extract_contextual_info(input_description)
        
        ranked_method_names = self.pne.evaluate_method_names(best_examples['Method Name'].tolist(), input_description)
        insights = self.gir.generate_all_insights(ranked_method_names, contextual_info["Entities"], contextual_info["Actions"], contextual_info["Context Scope"])
        
        prompt = "Best Examples\n"
        for example in best_examples.itertuples():
            prompt += f"\nFunctional Description: {example._1}, Method Name: {example._2}"
        
        prompt += "\nContextual Information:\n"
        prompt += f"Entities: {', '.join(contextual_info['Entities'])}, Actions: {', '.join(contextual_info['Actions'])}, Context Scope: {contextual_info['Context Scope']}"
        
        prompt += "\nRanked Potential Names:\n"
        for method_name, _, _, _ in ranked_method_names:
            prompt += f"({method_name}, Reason Insight: {insights.pop(0)})"
        
        prompt += "\nOther Examples ….\n"
        prompt += f"Query: {input_description}"
        
        return prompt
