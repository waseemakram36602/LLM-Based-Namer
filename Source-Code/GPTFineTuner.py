import openai

class GPTFineTuner:
    def __init__(self, api_key, model='gpt-3.5-turbo'):
        """
        Initialize the GPTFineTuner with OpenAI API key and model name.

        :param api_key: OpenAI API key.
        :param model: Name of the GPT model to use for fine-tuning (default: 'gpt-3.5-turbo').
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key

    def upload_corpus(self, corpus_file):
        """
        Upload the SFT training corpus to OpenAI's servers.

        :param corpus_file: Path to the SFT training corpus in JSONL format.
        :return: File ID of the uploaded corpus.
        """
        with open(corpus_file, 'r') as f:
            response = openai.File.create(file=f, purpose='fine-tune')
        return response['id']

    def fine_tune(self, file_id):
        """
        Fine-tune the GPT model using the uploaded corpus.

        :param file_id: File ID of the uploaded corpus.
        :return: Fine-tune job ID.
        """
        response = openai.FineTune.create(training_file=file_id, model=self.model)
        return response['id']

    def monitor_fine_tuning(self, job_id):
        """
        Monitor the fine-tuning process until completion.

        :param job_id: Fine-tune job ID.
        """
        while True:
            response = openai.FineTune.retrieve(job_id)
            status = response['status']
            if status in ['succeeded', 'failed']:
                break
            print(f"Fine-tuning status: {status}")
            time.sleep(60)
        return status
