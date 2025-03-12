# Third-party imports
from keras.api.models import Model  # Model class
from keras.api.layers import Input, Dense  # Layers for building the model
from keras.api.optimizers import Adam, SGD  # Optimizers for training
from keras.api.losses import MeanSquaredError  # Loss function for regression
from keras.api.regularizers import l2


# Function to create a regression model
def build_model(model_number: int) -> Model:
    """
    Returns a compiled regression model based on the specified model number.

    Parameters:
    - model_number (int): Model variant to create (1 to 5).

    Returns:
    - Compiled model.
    """

    print("\n🎯 Build Model 🎯\n")

    # Select model architecture and compile it
    if model_number == 1:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=4, activation="relu")(input_layer)
        output_layer = Dense(units=1)(first_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m1")
        model.compile(optimizer=Adam(), loss=MeanSquaredError())

    elif model_number == 2:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=8, activation="relu")(input_layer)
        output_layer = Dense(units=1)(first_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m2")
        model.compile(optimizer=Adam(), loss=MeanSquaredError())

    elif model_number == 3:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=8, activation="relu")(input_layer)
        second_layer = Dense(units=4, activation="relu")(first_layer)
        output_layer = Dense(units=1)(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m3")
        model.compile(optimizer=Adam(), loss=MeanSquaredError())

    elif model_number == 4:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=8, activation="relu")(input_layer)
        second_layer = Dense(units=4, activation="relu")(first_layer)
        output_layer = Dense(units=1)(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m4")
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    elif model_number == 5:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=8, activation="relu")(input_layer)
        second_layer = Dense(units=4, activation="relu")(first_layer)
        output_layer = Dense(units=1)(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m5")
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss=MeanSquaredError())


    elif model_number == 6:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=13, activation="relu")(input_layer)
        second_layer = Dense(units=8, activation="relu")(first_layer)
        third_layer = Dense(units=4, activation="relu")(second_layer)
        output_layer = Dense(units=1)(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m6")
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    elif model_number == 7:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=13, activation="sigmoid")(input_layer)
        second_layer = Dense(units=8, activation="sigmoid")(first_layer)
        third_layer = Dense(units=4, activation="sigmoid")(second_layer)
        output_layer = Dense(units=1)(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m7")
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    elif model_number == 8:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=13, activation="relu")(input_layer)
        second_layer = Dense(units=8, activation="relu")(first_layer)
        third_layer = Dense(units=4, activation="relu")(second_layer)
        output_layer = Dense(units=1, activation="sigmoid")(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m8")
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    elif model_number == 9:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=25, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(input_layer)
        second_layer = Dense(units=15, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(first_layer)
        third_layer = Dense(units=8, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(second_layer)
        output_layer = Dense(units=1, activation="sigmoid", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m9")
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    elif model_number == 10:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=25, activation="relu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(input_layer)
        second_layer = Dense(units=15, activation="relu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(first_layer)
        third_layer = Dense(units=8, activation="relu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(second_layer)
        output_layer = Dense(units=1, activation="sigmoid", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m10")
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    elif model_number == 11:
        input_layer = Input(shape=(13,))
        first_layer = Dense(units=25, activation="relu", kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(input_layer)
        second_layer = Dense(units=15, activation="relu", kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(first_layer)
        third_layer = Dense(units=8, activation="relu", kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(second_layer)
        output_layer = Dense(units=1, activation="sigmoid", kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(second_layer)
        model = Model(inputs=input_layer, outputs=output_layer, name="m10")
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())


    else:
        raise ValueError(f"❌ Invalid model number: {model_number}")

    # Display model summary (after compilation)
    model.summary()

    return model

# Print confirmation message
print("\n✅ model.py successfully executed")