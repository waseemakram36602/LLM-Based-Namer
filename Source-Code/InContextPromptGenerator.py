class InContextPromptGenerator:
    def __init__(self, optimal_prompt_template):
        """
        Initialize the InContextPromptGenerator with the optimal prompt template.

        :param optimal_prompt_template: The optimal prompt template selected based on highest AvgNameSim.
        """
        self.optimal_prompt_template = optimal_prompt_template

    def generate_in_context_prompt(self, input_description, best_examples):
        """
        Generate an in-context prompt by integrating the input description and best examples into the optimal prompt template.

        :param input_description: The input functional description.
        :param best_examples: List of best examples [(functional_description, method_name, similarity_score), ...]
        :return: The generated in-context prompt.
        """
        examples_section = ""
        for example in best_examples:
            example_description = example[0]
            example_method_name = example[1]
            examples_section += f"Example:\nFunctional Description: {example_description}\nMethod Name: {example_method_name}\n\n"

        in_context_prompt = self.optimal_prompt_template.replace("{functional_description}", input_description)
        in_context_prompt = in_context_prompt.replace("{examples_section}", examples_section.strip())

        return in_context_prompt
