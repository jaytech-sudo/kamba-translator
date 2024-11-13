import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import os
import pickle  # For saving tokenizers

# Define paths
TRAINING_DATA_PATH = r'c:/Users/vivia/kamba_translation_api/cleaned_file.xlsx'
MODEL_SAVE_PATH = "translation_model.keras"
TOKENIZER_INPUT_PATH = "tokenizer_input.pkl"
TOKENIZER_OUTPUT_PATH = "tokenizer_output.pkl"

class TranslationModel:
    def __init__(self):
        self.tokenizer_input = Tokenizer()
        self.tokenizer_output = Tokenizer()
        self.model = None
        self.max_input_length = None
        self.max_output_length = None

    def load_data(self):
        data = pd.read_excel(TRAINING_DATA_PATH)
        self.input_texts = data['kamba'].astype(str).tolist()
        self.output_texts = data['english'].astype(str).tolist()
        print("Data loaded successfully.")

    def preprocess_data(self):
        # Fit tokenizers on text data
        self.tokenizer_input.fit_on_texts(self.input_texts)
        self.tokenizer_output.fit_on_texts(self.output_texts)

        input_sequences = self.tokenizer_input.texts_to_sequences(self.input_texts)
        output_sequences = self.tokenizer_output.texts_to_sequences(self.output_texts)

        self.max_input_length = max(len(seq) for seq in input_sequences)
        self.max_output_length = self.max_input_length
        self.input_sequences = pad_sequences(input_sequences, maxlen=self.max_input_length, padding='post')
        self.output_sequences = pad_sequences(output_sequences, maxlen=self.max_output_length, padding='post')

        self.input_vocab_size = len(self.tokenizer_input.word_index) + 1
        self.output_vocab_size = len(self.tokenizer_output.word_index) + 1

        print("Input vocabulary size:", self.input_vocab_size)
        print("Output vocabulary size:", self.output_vocab_size)
        print("Max input length:", self.max_input_length)
        print("Max output length:", self.max_output_length)
        print("Data preprocessing completed.")

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.input_vocab_size, 256, input_length=self.max_input_length))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dense(self.output_vocab_size, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        print("Model built successfully.")

    def train_model(self, epochs=300, batch_size=64):
        self.output_sequences = np.expand_dims(self.output_sequences, -1)
        print(f"Shape of input sequences: {self.input_sequences.shape}")
        print(f"Shape of output sequences: {self.output_sequences.shape}")
        self.model.fit(self.input_sequences, self.output_sequences, epochs=epochs, batch_size=batch_size)
        print("Model training completed.")

    def save_model(self):
        # Save the trained model
        self.model.save(MODEL_SAVE_PATH)
        print("Model saved successfully.")

        # Save tokenizers to files for later use
        with open(TOKENIZER_INPUT_PATH, 'wb') as f:
            pickle.dump(self.tokenizer_input, f)
        with open(TOKENIZER_OUTPUT_PATH, 'wb') as f:
            pickle.dump(self.tokenizer_output, f)
        print("Tokenizers saved successfully.")

    def load_model(self):
        if os.path.exists(MODEL_SAVE_PATH):
            self.model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            print("Model loaded successfully.")
        else:
            print("Model file not found. Please train and save the model first.")

    def translate(self, text):
        if self.model is None:
            return "Translation model not loaded. Please train or load a model first."

        print("Input text for translation:", text)
        sequence = self.tokenizer_input.texts_to_sequences([text])
        print("Tokenized input sequence:", sequence)
        
        if not sequence or not sequence[0]:  # Check if the sequence is empty
            return "Translation not possible: Invalid input text."

        padded_sequence = pad_sequences(sequence, maxlen=self.max_input_length, padding='post')
        print("Padded input sequence:", padded_sequence)

        prediction = self.model.predict(padded_sequence)
        
        print("Raw prediction output:", prediction)
        
        predicted_sequence = np.argmax(prediction, axis=-1)[0]
        print("Predicted translation sequence:", predicted_sequence)

        # Filter out zeros
        output_text = ' '.join(
            [self.tokenizer_output.index_word.get(index, '') for index in predicted_sequence if index != 0]
        )
        print("Filtered Predicted translation:", output_text)

        if not output_text:
            return "Translation incomplete: insufficient data or vocabulary coverage."

        return output_text.strip()


# Example usage
if __name__ == "__main__":
    translation_model = TranslationModel()
    translation_model.load_data()
    translation_model.preprocess_data()
    translation_model.build_model()
    
    # Uncomment these lines to train and save the model
    # translation_model.train_model(epochs=5)
    # translation_model.save_model()
    
    translation_model.load_model()
    
    # Example translation
    translated_text = translation_model.translate("sumuni")
    print("Translated text:", translated_text)
