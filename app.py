from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# Define paths
MODEL_SAVE_PATH = "translation_model.keras"
TOKENIZER_INPUT_PATH = "tokenizer_input.pkl"
TOKENIZER_OUTPUT_PATH = "tokenizer_output.pkl"

class TranslationModel:
    def __init__(self):
        self.tokenizer_input = None
        self.tokenizer_output = None
        self.model = None
        self.max_input_length = None

    def load_tokenizers(self):
        # Load saved tokenizers
        with open(TOKENIZER_INPUT_PATH, 'rb') as f:
            self.tokenizer_input = pickle.load(f)
        with open(TOKENIZER_OUTPUT_PATH, 'rb') as f:
            self.tokenizer_output = pickle.load(f)
        print("Tokenizers loaded successfully.")

    def load_model(self):
        if os.path.exists(MODEL_SAVE_PATH):
            self.model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            print("Model loaded successfully.")
        else:
            print("Model file not found. Please train and save the model first.")

    def translate(self, text):
        if self.model is None:
            return "Translation model not loaded. Please train or load a model first."

        # Tokenize input text
        print("Input text for translation:", text)
        sequence = self.tokenizer_input.texts_to_sequences([text])
        print("Tokenized input sequence:", sequence)

        if not sequence or not sequence[0]:  # Check if the sequence is empty
            return "Translation not possible: Unknown input text."

        padded_sequence = pad_sequences(sequence, maxlen=self.max_input_length, padding='post')
        print("Padded input sequence:", padded_sequence)

        # Predict translation
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


# Create translation model instance
translation_model = TranslationModel()
translation_model.load_tokenizers()
translation_model.load_model()

# Flask endpoint for translation
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    kamba_text = data.get('kamba_text')

    if not kamba_text:
        return jsonify({"error": "No Kamba text provided"}), 400
    
    try:
        # Translate Kamba text to English
        english_translation = translation_model.translate(kamba_text)
        if not english_translation:
            return jsonify({"error": "Translation failed or returned empty result"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        'kamba_text': kamba_text,
        'english_translation': english_translation
    })

if __name__ == '__main__':
    app.run(debug=True)
