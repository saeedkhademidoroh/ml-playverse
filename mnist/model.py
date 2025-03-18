# Third-party imports
from keras.api.models import Model
from keras.api.layers import Input, Dense
from keras.api.optimizers import Adam
from keras.api.losses import CategoricalCrossentropy


# Function to create model
def build_model(model_number: int) -> Model:
    """
    Returns compiled model based on specified model number.

    Parameters:
    - model_number (int): Model variant to create (1 to 5).

    Returns:
    - Compiled model and description (if any).
    """


    print("\n🎯 Build Model 🎯\n")

    # Select model architecture and compile it
    if model_number == 1:
        input_layer = Input(shape=(784,))
        first_layer = Dense(units=512, activation="relu")(input_layer)
        second_layer = Dense(units=256, activation="relu")(first_layer)
        third_layer = Dense(units=128, activation="relu")(second_layer)
        output_layer = Dense(units=10, activation="softmax")(third_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m1")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        description = "1. Input layer has shape of 784 (28x28 pixels).\n"
        description = description + "2. Hidden layers have 512, 256, and 128 units and ReLU activation.\n"
        description = description + "3. Output layer has 10 units with softmax activation.\n"
        description = description + "4. Compiled with Adam optimizer and Categorical Cross Entropy loss function."

    elif model_number == 2:
        print(f"❌ Invalid model number: {model_number}")

    else:
        raise ValueError(f"❌ Invalid model number: {model_number}")

    # Display model summary (after compilation)
    model.summary()

    return model, description


# Print confirmation message
print("\n✅ model.py successfully executed")