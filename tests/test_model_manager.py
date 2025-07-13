import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ptycho.model_manager import ModelManager

def test_save_and_load_model():
    # Create a simple model for testing
    model = Sequential([
        Dense(64, activation='relu', input_shape=(32,)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define custom objects and intensity scale for testing
    custom_objects = {'custom_activation': tf.nn.relu}
    intensity_scale = 2.5

    # Save the model
    model_path = 'test_model.h5'
    ModelManager.save_model(model, model_path, custom_objects, intensity_scale)

    # Ensure the .dill file is created
    assert os.path.exists(model_path + ".dill")

    # Load the model
    loaded_model = ModelManager.load_model(model_path)

    # Check if the loaded model has the same architecture
    assert np.array_equal(model.get_weights()[0], loaded_model.get_weights()[0])

    # Clean up
    os.remove(model_path)
    os.remove(model_path + ".dill")
