alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}  # CTC: blank = 0
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

def text_to_indices(text):
    return [char_to_idx[char] for char in text if char in char_to_idx]

def indices_to_text(indices):
    return ''.join([idx_to_char.get(idx, '') for idx in indices])

def ctc_decode(logits):
    preds = logits.argmax(2)
    pred_texts = []
    for pred in preds:
        pred = pred.detach().cpu().numpy()
        no_dup = []
        prev = -1
        for p in pred:
            if p != prev and p != 0:
                no_dup.append(p)
            prev = p
        pred_texts.append(indices_to_text(no_dup))
    return pred_texts
