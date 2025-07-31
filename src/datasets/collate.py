import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    data_objects = [item["data_object"] for item in dataset_items]
    result_batch["data_object"] = torch.stack(data_objects)

    result_batch["labels"] = torch.tensor([item["labels"] for item in dataset_items])
    return result_batch
