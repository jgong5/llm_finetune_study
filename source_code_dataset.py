import datasets
import json
import os

class SourceCodeDataset(datasets.GeneratorBasedBuilder):
    """
    A huggingface dataset for source files
    Given a json file containing a list of file paths, the code builds a dataset with the following features:
    1. "file_path": relative path to the project base dir.
    2. "code": the code content of the file.
    """

    def __init__(self, json_file: str, **kwargs):
        super().__init__(**kwargs)
        with open(json_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.project_base = config["project_base"]
        self.file_paths = config["file_paths"]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "file_path": datasets.Value("string"), # The relative path of the file
                    "code": datasets.Value("string"), # The content of the file
                }
            )
        )

    # Define how to split the dataset into train, test, and validation
    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # In this case, we assume that the base dir of the project is already available locally
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
            ),
        ]

    # Define how to generate examples from the data
    def _generate_examples(self):
        # This method handles the data generation from the local files
        # It yields tuples of (key, example), where key is a unique id and example is a dictionary of features
        # Iterate over all the files in the project directory
        for file_path in self.file_paths:
            # Read the content of the file
            with open(os.path.join(self.project_base, file_path), "r", encoding="utf-8") as f:
                code = f.read()
            # Yield a key and an example
            yield file_path, {
                "file_path": file_path,
                "code": code,
            }