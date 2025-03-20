"""finance: A Flower / FlowerTune app."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

FDS = None  # Cache FederatedDataset
GEMMA_MODEL_NAME = "google/gemma-3-4b-it" 


def formatting_prompts_func(example):
    """Construct prompts."""
    output_texts = []
    mssg = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    for i in range(len(example["instruction"])):
        text = (
            f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n"
            f"### Response: {example['response'][i]}"
        )
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(
    model_name: str = GEMMA_MODEL_NAME,
):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )
    return tokenizer, data_collator, formatting_prompts_func


def formatting(dataset):
    """Format dataset."""
    if "input" in dataset and dataset["input"][0]:
        dataset["instruction"] = dataset["instruction"] + " " + dataset["input"]
        dataset = dataset.remove_columns(["input"])
    return dataset


def reformat(dataset, llm_task):
    """Reformat datasets."""
    # Handle the different column names from the dataset
    if "conversations" in dataset.column_names:

        def extract_instruction_response(example):
            instruction = example["conversations"][0]["value"]
            response = example["conversations"][1]["value"]
            return {"instruction": instruction, "response": response}

        dataset = dataset.map(
            extract_instruction_response, remove_columns=["conversations"]
        )

    if llm_task in ["finance", "code"]:
        dataset = dataset.map(formatting)
    if llm_task == "medical":
        if "instruction" in dataset.column_names:
            dataset = dataset.remove_columns(["instruction"])
        dataset = dataset.rename_column("input", "instruction")
    return dataset


def load_data(
    partition_id: int, num_partitions: int, dataset_name: str = "mlabonne/FineTome-100k"
):
    """Load partition data."""
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")
    client_trainset = reformat(client_trainset, llm_task="finance")
    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


def main():
    """Main function to demonstrate usage."""
    num_partitions = 10  # Example number of partitions
    partition_id = 0  # Example partition ID
    dataset_name = "mlabonne/FineTome-100k"

    # Load data for a specific partition
    train_dataset = load_data(partition_id, num_partitions, dataset_name)

    # Get tokenizer and data collator
    tokenizer, data_collator, prompt_formatting_func = (
        get_tokenizer_and_data_collator_and_propt_formatting()
    )

    # Example: Print the first few examples from the dataset
    print("Example dataset entries:")
    for i in range(min(5, len(train_dataset))):
        print(train_dataset[i])

    print("\nExample formatted prompts:")
    formatted_prompts = prompt_formatting_func(train_dataset[:1])
    for prompt in formatted_prompts:
        print(prompt)


if __name__ == "__main__":
    main()
