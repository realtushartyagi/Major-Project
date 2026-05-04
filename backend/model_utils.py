import json
import os

# Heavy imports moved inside

MAX_LEN = 200
VOCAB_SIZE = 100

def build_model(vocab_size, embedding_dim=32, max_len=200):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class URLTokenizer:
    def __init__(self, vocab_size=100, max_len=200):
        from tensorflow.keras.preprocessing.text import Tokenizer
        self.tokenizer = Tokenizer(num_words=vocab_size, char_level=True, lower=True)
        self.max_len = max_len

    def fit(self, urls):
        self.tokenizer.fit_on_texts(urls)

    def transform(self, urls):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        sequences = self.tokenizer.texts_to_sequences(urls)
        return pad_sequences(sequences, maxlen=self.max_len)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.tokenizer.to_json())

    @classmethod
    def load(cls, path, max_len=200):
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        with open(path, 'r') as f:
            json_string = f.read()
        obj = cls(max_len=max_len)
        obj.tokenizer = tokenizer_from_json(json_string)
        return obj
