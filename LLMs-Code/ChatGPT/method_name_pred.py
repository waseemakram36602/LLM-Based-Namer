# method_name_pred.py

from chatgpt4_config import ChatGPT4Configuration
from text_processing import TextProcessing
from csv_handler import CSVHandler

class MethodNamePrediction:
    def __init__(self, api_key):
        """Initializes the MethodNamePrediction class with the provided API key."""
        self.config = ChatGPT4Configuration(api_key)
    
    def process_prompts_from_csv(self, input_csv_file, output_csv_file):
        """Processes the prompts from a CSV file and generates outputs."""
        # Read the prompts from the CSV file
        prompts_df = CSVHandler.read_prompts_from_csv(input_csv_file)

        # Generate output for each prompt
        text_processor = TextProcessing(self.config)
        prompts_df['output'] = prompts_df['prompt'].apply(text_processor.generate_output_for_prompt)

        # Save the result to an output CSV file
        CSVHandler.save_output_to_csv(prompts_df, output_csv_file)
