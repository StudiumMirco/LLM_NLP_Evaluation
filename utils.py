import json


def parse_messages_for_gemini(data):
    """
    Parses the given input data by replacing the key "content" with "parts".

    Args:
        data (list): A list of dictionaries, each containing the keys "role" and "content".

    Returns:
        list: A list of dictionaries with the keys "role" and "parts".
    """
    parsed_data = []  # Initialize an empty list to store the parsed messages
    for item in data:
        # Create a new dictionary for each message with updated keys
        new_item = {
            # If the original role is "assistant", change it to "model", otherwise keep it unchanged
            "role": "model" if item["role"] == "assistant" else item["role"],
            # Replace the key "content" with "parts" by removing "content" from the original dict
            "parts": item.pop("content")
        }
        parsed_data.append(new_item)  # Add the updated dictionary to the parsed_data list
    return parsed_data


# Helper method to avoid frequent API calls.
# This method reads predictions and references from a JSON file.
def load_predictions_and_references(filepath):
    """
    Loads predictions and references from a JSON file.

    Args:
        filepath (str): The path to the JSON file containing predictions and references.

    Returns:
        tuple: A tuple containing a list of predictions and a list of references.
    """
    with open(filepath, "r") as file:
        data = json.load(file)  # Parse the JSON data from the file

    # Extract the "predictions" and "references" keys from the data, using an empty list as a default
    predictions = data.get("predictions", [])
    references = data.get("references", [])
    return predictions, references


def load_json(filepath):
    """
    Loads a JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        list: The parsed JSON data.
    """
    with open(filepath, "r") as file:
        data = json.load(file)  # Read and parse the JSON data from the file
    return data


def save_json(data, file_path):
    """
    Saves the provided data to a JSON file.

    Args:
        data: The data to save in JSON format.
        file_path (str): The path where the JSON file will be saved.
    """
    # Write the JSON data to the file with pretty-print formatting and UTF-8 encoding
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
