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
    # Items
    audios = [item["audio"].squeeze(0).T for item in dataset_items]
    specs = [item["spectrogram"].T for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    text_encoded = [torch.tensor(item["text_encoded"], dtype=torch.long) for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    # Padding
    audio_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    spec_padded = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
    text_encoded_padded = torch.nn.utils.rnn.pad_sequence(text_encoded, batch_first=True, padding_value=0)

    # Lengths
    spec_lengths = torch.tensor([s.shape[0] for s in specs], dtype=torch.long)
    text_lengths = torch.tensor([len(t) for t in text_encoded], dtype=torch.long)

    result_batch = {
        "audio": audio_padded,
        "spectrogram": spec_padded,
        "text": texts,
        "text_encoded": text_encoded_padded,
        "audio_path": audio_paths,
        "spectrogram_lengths": spec_lengths,
        "text_encoded_lengths": text_lengths,
    }

    return result_batch
