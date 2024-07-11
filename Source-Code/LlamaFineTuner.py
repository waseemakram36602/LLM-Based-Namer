from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import datasets

class LlamaFineTuner:
    def __init__(self, model_name='llama-3', output_dir='./results'):
        """
        Initialize the LlamaFineTuner with model name and output directory.

        :param model_name: Name of the Llama model to use for fine-tuning (default: 'llama-3').
        :param output_dir: Directory to save the fine-tuned model.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def load_dataset(self, corpus_file):
        """
        Load the SFT training corpus as a dataset.

        :param corpus_file: Path to the SFT training corpus in JSONL format.
        :return: Loaded dataset.
        """
        dataset = datasets.load_dataset('json', data_files=corpus_file)
        return dataset

    def preprocess_function(self, examples):
        """
        Preprocess the dataset examples.

        :param examples: Dataset examples.
        :return: Preprocessed examples.
        """
        inputs = examples['messages']
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        return model_inputs

    def fine_tune(self, dataset):
        """
        Fine-tune the Llama model using the provided dataset.

        :param dataset: Dataset for fine-tuning.
        """
        tokenized_datasets = dataset.map(self.preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train']
        )
        trainer.train()


