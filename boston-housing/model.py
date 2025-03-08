# Import necessary libraries
from keras.api.models import Model  # Model class
from keras.api.layers import Input, Dense  # Layers for the model
from keras.api.optimizers import Adam, SGD  # Optimizer for the model
from keras.api.losses import MeanSquaredError  # Loss function for the model
from keras.api.callbacks import EarlyStopping  # Callback for the model

# Function to create a regression model
def get_model(model_number: int) -> Model:
    """
    Returns a compiled regression model based on the specified model number.

    Parameters:
    - model_number (int): Model variant to create (1 to 5).

    Returns:
    - Compiled Keras Model.
    """

    print("\n🎯 Regression Model Creation 🎯\n")

    # Define input layer (common for all models)
    input_layer = Input(shape=(13,))

    # Select model architecture and compile it
    if model_number == 1:
        first_layer = Dense(units=4, activation="relu")(input_layer)
        output_layer = Dense(units=1)(first_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m1")
        model.compile(optimizer=Adam(), loss=MeanSquaredError())

    elif model_number == 2:
        first_layer = Dense(units=8, activation="relu")(input_layer)
        output_layer = Dense(units=1)(first_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m2")
        model.compile(optimizer=Adam(), loss=MeanSquaredError())

    elif model_number == 3:
        first_layer = Dense(units=8, activation="relu")(input_layer)
        second_layer = Dense(units=4, activation="relu")(first_layer)
        output_layer = Dense(units=1)(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m3")
        model.compile(optimizer=Adam(), loss=MeanSquaredError())

    elif model_number == 4:
        first_layer = Dense(units=8, activation="relu")(input_layer)
        second_layer = Dense(units=4, activation="relu")(first_layer)
        output_layer = Dense(units=1)(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m4")
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss=MeanSquaredError())

    elif model_number == 5:
        first_layer = Dense(units=8, activation="relu")(input_layer)
        second_layer = Dense(units=4, activation="relu")(first_layer)
        output_layer = Dense(units=1)(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m5")
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    else:
        raise ValueError(f"❌ Invalid model number: {model_number}")

    # Display model summary (after compilation)
    model.summary()

    return model

# Print confirmation message
print("\n✅ model.py successfully executed\n")