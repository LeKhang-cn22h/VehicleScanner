# charset.py
class LabelConverter:
    def __init__(self, chars):
        self.chars = ['[blank]'] + list(chars)
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def encode(self, text):
        return [self.char2idx[char] for char in text]

    def decode(self, preds):
        pred_str = []
        for i in range(len(preds)):
            if preds[i] != 0 and (i == 0 or preds[i] != preds[i - 1]):
                pred_str.append(self.idx2char[preds[i]])
        return ''.join(pred_str)
