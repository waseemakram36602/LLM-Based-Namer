import json

class SFTCorpusFormation:
    def __init__(self, tuning_dataset):
        """
        Initialize the SFTCorpusFormation with a tuning dataset.

        :param tuning_dataset: List of tuples [(functional_description, method_name), ...]
        """
        self.tuning_dataset = tuning_dataset

    def create_corpus_entry(self, functional_description, method_name):
        """
        Create a single entry for the SFT corpus.

        :param functional_description: The functional description of the method.
        :param method_name: The corresponding method name.
        :return: A dictionary representing the JSON entry.
        """
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "Suggest a clear, convention-compliant method name based on the given functional description."
                },
                {
                    "role": "user",
                    "content": functional_description
                },
                {
                    "role": "assistant",
                    "content": method_name
                }
            ]
        }
        return entry

    def create_corpus(self):
        """
        Create the SFT corpus from the tuning dataset.

        :return: A list of JSON entries representing the SFT corpus.
        """
        corpus = []
        for functional_description, method_name in self.tuning_dataset:
            entry = self.create_corpus_entry(functional_description, method_name)
            corpus.append(entry)
        return corpus

    def save_corpus_to_file(self, filename):
        """
        Save the SFT corpus to a JSON file.

        :param filename: The name of the file to save the corpus to.
        """
        corpus = self.create_corpus()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
