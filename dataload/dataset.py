import os
import json

from datasets import load_dataset


# Base Dataset class (Inspired by promptbench)
class Dataset(object):
    """
    A base class for datasets.

    This class provides basic functionality for loading and accessing dataset data.
    """

    def __init__(self, dataset_name):
        self.data = []  # List to hold dataset entries.
        self.dataset_name = dataset_name

        # Determine the data directory (parent directory's 'data' folder)
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(cur_dir), 'data')
        # Create the data directory if it does not exist.
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        assert len(self.data) > 0, "Empty dataset. Please load data first."
        return len(self.data)

    def __getitem__(self, idx):
        """
        Allows access to the dataset entries via indices.
        """
        assert len(self.data) > 0, "Empty dataset. Please load data first."
        return self.data[idx]

    def extract_answer(self, output):
        """
        Placeholder method for extracting an answer from model output.
        """
        return output


# SQUAD_V2 dataset for question answering tasks.
class SQUAD_V2(Dataset):
    """
    SQUAD_V2 is a dataset class for the Stanford Question Answering Dataset (SQuAD) version 2.
    The dataset is loaded from the Hugging Face datasets library.

    Example data format:
    [{'id': '56ddde6b9a695914005b9628', 'title': 'Normans', 'context': '...',
      'question': 'In what country is Normandy located?',
      'answers': {'text': ['France', 'France', ...], 'answer_start': [159, 159, ...]}}, ...]
    """

    def __init__(self):
        self.data = []
        # Load the 'validation' split of the squad_v2 dataset.
        data = load_dataset("squad_v2")["validation"]
        for d in data:
            self.data.append(d)


# SQUAD_V2 Question Generation dataset.
class SQUAD_V2_QG(Dataset):
    """
    SQUAD_V2_QG is a dataset class for question generation based on the SQuAD_v2 dataset.
    This version is loaded from the GEM/squad_v2 dataset on Hugging Face.
    """

    def __init__(self):
        self.data = []
        data = load_dataset("GEM/squad_v2")["validation"]
        for d in data:
            self.data.append(d)


# CNN/DailyMail dataset for summarization tasks.
class CNNDailyMail(Dataset):
    """
    CNNDailyMail dataset for summarization tasks.
    """

    def __init__(self):
        # Call the parent constructor with dataset name.
        super().__init__("cnn_dailymail")
        # Load the 'validation' split of the CNN/DailyMail dataset (version 3.0.0).
        data = load_dataset("cnn_dailymail", "3.0.0", split="validation")
        for d in data:
            self.data.append(d)


# ROCStories dataset for text generation tasks.
class ROCStories(Dataset):
    """
    ROCStories dataset for story or text generation tasks.
    """

    def __init__(self):
        super().__init__("rocstories")
        # Load the 'validation' split of the ROCStories dataset.
        data = load_dataset("Ximing/ROCStories", split="validation")
        for d in data:
            self.data.append(d)


# WMT16 dataset for machine translation tasks.
class WMT16(Dataset):
    """
    WMT16 dataset for machine translation tasks.
    """

    def __init__(self):
        super().__init__("wmt16")
        # Load the 'validation' split of the WMT16 dataset for German-English translation.
        data = load_dataset("wmt16", "de-en", split="validation")
        for d in data:
            self.data.append(d)


# OPUS100 dataset for machine translation tasks.
class OPUS100(Dataset):
    """
    OPUS100 dataset for machine translation tasks.
    """

    def __init__(self):
        super().__init__("opus100")
        # Load the 'validation' split of the OPUS100 dataset for German-English translation.
        data = load_dataset("Helsinki-NLP/opus-100", "de-en", split="validation")
        for d in data:
            self.data.append(d)


# TopicalChat dataset for dialogue tasks.
class TopicalChat(Dataset):
    """
    TopicalChat dataset for dialogue tasks.
    """

    def __init__(self):
        super().__init__("topical_chat")
        # Construct the path to the TopicalChat validation data stored as JSON.
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(cur_dir), 'data\\topical_chat_valid_freq.json')
        with open(data_dir, "r") as file:
            data = json.load(file)
        for d in data:
            self.data.append(d)


# MTBENCH dataset for evaluating writing or dialog tasks.
class MTBENCH(Dataset):
    """
    MTBENCH dataset for machine translation benchmark tasks.
    """

    def __init__(self):
        super().__init__("mtbench")
        # Construct the path to the MTBENCH questions data (assumed to be in JSONL format).
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(cur_dir), 'data\\questions_mtbench.jsonl')
        with open(data_dir, "r") as file:
            data = json.load(file)
        for d in data:
            self.data.append(d)


# General SuperGLUE Dataset class for tasks like CB, COPA, RTE, etc.
class SuperGLUE_Dataset(Dataset):
    """
    A general dataset class for SuperGLUE tasks, to be extended for specific tasks (e.g., CB, COPA).
    """

    def __init__(self, dataset_name, subset):
        super().__init__(dataset_name)
        self.subset = subset  # Subset identifier (e.g., "cb", "copa")
        self.load_data()

    def load_data(self):
        # Load the 'validation' split of the specified SuperGLUE dataset subset.
        data = load_dataset("super_glue", self.subset, trust_remote_code=True)["validation"]
        for d in data:
            self.data.append(d)


# SuperGLUE CB task dataset.
class CB(SuperGLUE_Dataset):
    """
    SuperGLUE CB task for recognizing contrast.
    """

    def __init__(self):
        super().__init__("super_glue", "cb")


# SuperGLUE Boolq task dataset.
class Boolq(SuperGLUE_Dataset):
    """
    Loads the Boolq dataset from SuperGLUE.
    """

    def __init__(self):
        super().__init__("super_glue", "boolq")


# SuperGLUE AXB task dataset.
class AXB(SuperGLUE_Dataset):
    """
    Loads the AXB dataset from SuperGLUE.
    """

    def __init__(self):
        super().__init__("super_glue", "axb")

    def load_data(self):
        data = load_dataset("super_glue", self.subset, trust_remote_code=True)["test"]
        for d in data:
            self.data.append(d)


# SuperGLUE AXG task dataset.
class AXG(SuperGLUE_Dataset):
    """
    Loads the AXG dataset from SuperGLUE.
    """

    def __init__(self):
        super().__init__("super_glue", "axg")

    def load_data(self):
        # Load the 'test' split for the AXG dataset.
        data = load_dataset("super_glue", self.subset, trust_remote_code=True)["test"]
        for d in data:
            self.data.append(d)


# SuperGLUE COPA task dataset.
class COPA(SuperGLUE_Dataset):
    """
    Loads the COPA dataset from SuperGLUE.
    """

    def __init__(self):
        super().__init__("super_glue", "copa")


# SuperGLUE MultiRC task dataset.
class MultiRC(SuperGLUE_Dataset):
    """
    Loads the MultiRC dataset from SuperGLUE.
    """

    def __init__(self):
        super().__init__("super_glue", "multirc")


# SuperGLUE ReCoRD task dataset.
class RECORD(SuperGLUE_Dataset):
    """
    Loads the ReCoRD dataset from SuperGLUE.
    """

    def __init__(self):
        super().__init__("super_glue", "record")


# SuperGLUE RTE task dataset.
class RTE(SuperGLUE_Dataset):
    """
    Loads the RTE dataset from SuperGLUE.
    """

    def __init__(self):
        super().__init__("super_glue", "rte")


# SuperGLUE WiC task dataset.
class WIC(SuperGLUE_Dataset):
    """
    Loads the WiC dataset from SuperGLUE.
    """

    def __init__(self):
        super().__init__("super_glue", "wic")


# SuperGLUE WSC task dataset.
class WSC(SuperGLUE_Dataset):
    """
    Loads the WSC dataset from SuperGLUE.
    """

    def __init__(self):
        super().__init__("super_glue", "wsc")
