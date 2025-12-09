import torch
from torch.nn.utils.rnn import pad_sequence


def sasrec_collate_fn(dataset_items: list[dict], pad_token: int):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
        pad_token (int): index of padding token for sequences
        max_len (int): maximum length of a session
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    seqs = [elem["seq"].flip(0) for elem in dataset_items]
    users = [elem["user"] for elem in dataset_items]
    items = [elem.get("item", None) for elem in dataset_items]

    seqs = pad_sequence(seqs, batch_first=True, padding_value=pad_token)
    seqs = seqs.flip(1)

    attention_mask = (seqs != pad_token).to(torch.long)

    result_batch = {
        "seq": seqs,
        "attention_mask": attention_mask,
        "user": torch.tensor(users),
    }

    if items[0] is not None:
        result_batch["item"] = torch.tensor(items)

    return result_batch
