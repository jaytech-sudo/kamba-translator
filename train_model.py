from model import TranslationModel

# Create the TranslationModel instance
translation_model = TranslationModel()

# Load and preprocess the data
translation_model.load_data()
translation_model.preprocess_data()

# Build the model
translation_model.build_model()

# Train the model (this may take some time depending on data size)
translation_model.train_model()

# Save the trained model
translation_model.save_model()
