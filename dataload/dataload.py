from .dataset import *


class DatasetLoader:
    @staticmethod
    def load_dataset(dataset_name):
        """
        Load and return the specified dataset.

        This function acts as a factory method, returning the appropriate dataset object
        based on the provided dataset name. Note that for certain datasets,
        additional arguments may be required to specify the task or languages. Refer to each dataset's documentation
        for details.

        Args:
            dataset_name (str): The name of the dataset to load.
                For example:
                  - "squad_v2" for the Stanford Question Answering Dataset v2,
                  - "cnn_dailymail" for the CNN/DailyMail summarization dataset,
                  - "wmt16" for the WMT16 machine translation dataset (e.g., German-English),
                  - "topical_chat" for the TopicalChat dialogue dataset, etc.

        Returns:
            Dataset object corresponding to the given dataset_name.
            The returned dataset object is typically a list where each element is a dictionary containing
            the dataset's fields (e.g., id, context, question, answers, etc.).

        Raises:
            NotImplementedError: If the dataset_name does not correspond to any known or supported dataset.
        """
        if dataset_name == "squad_v2":
            return SQUAD_V2()
        elif dataset_name == "squad_v2_qg":
            return SQUAD_V2_QG()
        elif dataset_name == "rocstories":
            return ROCStories()
        elif dataset_name == "cnn_dailymail":
            return CNNDailyMail()
        elif dataset_name == "wmt16":
            return WMT16()
        elif dataset_name == "opus100":
            return OPUS100()
        elif dataset_name == "topical_chat":
            return TopicalChat()
        elif dataset_name == "mtbench":
            return MTBENCH()
        elif dataset_name == "cb":
            return CB()
        elif dataset_name == "boolq":
            return Boolq()
        elif dataset_name == "axb":
            return AXB()
        elif dataset_name == "axg":
            return AXG()
        elif dataset_name == "copa":
            return COPA()
        elif dataset_name == "multirc":
            return MultiRC()
        elif dataset_name == "record":
            return RECORD()
        elif dataset_name == "rte":
            return RTE()
        elif dataset_name == "wic":
            return WIC()
        elif dataset_name == "wsc":
            return WSC()
        else:
            # If the dataset name doesn't match any known datasets, raise an error.
            raise NotImplementedError(f"Dataset '{dataset_name}' is not supported.")
