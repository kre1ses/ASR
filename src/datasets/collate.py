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

    batch_size = len(dataset_items)

    spectrogram_freq = dataset_items[0]['spectrogram'].shape[1]
    lengths = [elem['spectrogram'].shape[2] for elem in dataset_items]
    max_spec_length = max(lengths)

    texts = [elem['text'] for elem in dataset_items]
    text_encoded_lengths = [elem['text_encoded'].shape[1] for elem in dataset_items]
    max_text_length = max(text_encoded_lengths)

    audio_paths = [elem['audio_path'] for elem in dataset_items]
    audio = [elem['audio'] for elem in dataset_items]

    batch_spectrogram = torch.zeros((batch_size, spectrogram_freq, max_spec_length))

    batch_spectrogram = torch.log(batch_spectrogram + 1e-5)

    batch_text_encoded = torch.zeros((batch_size, max_text_length))

    for i in range(batch_size):
        batch_spectrogram[i, :, :lengths[i]] = dataset_items[i]['spectrogram'][0]
        batch_text_encoded[i, :text_encoded_lengths[i]] = dataset_items[i]['text_encoded'][0]
    
    result = {
        'spectrogram': batch_spectrogram,
        'spectrogram_lengths': torch.tensor(lengths, dtype=torch.long),
        'text_encoded': batch_text_encoded,
        'text_encoded_length': torch.tensor(text_encoded_lengths, dtype=torch.long),
        'text': texts,
        'audio_path': audio_paths,
        'audio': audio,
    }
    
    return result
