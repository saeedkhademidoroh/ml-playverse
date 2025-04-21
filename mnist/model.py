# Third-party imports
from keras.api.models import Model
from keras.api.layers import Input, Dense, Dropout
from keras.api.optimizers import Adam, SGD
from keras.api.losses import CategoricalCrossentropy
from keras.api.regularizers import l2


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
        description = None

    elif model_number == 2:
        input_layer = Input(shape=(784,))
        x = Dense(units=512, activation="relu")(input_layer)
        x = Dropout(0.2)(x)
        x = Dense(units=256, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(units=128, activation="relu")(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(units=10, activation="softmax")(x)
        model = Model(inputs=input_layer, outputs=output_layer, name="m2")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        description = None

    elif model_number == 3:
        input_layer = Input(shape=(784,))
        x = Dense(units=512, activation="relu")(input_layer)
        x = Dropout(0.2)(x)
        x = Dense(units=256, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(units=128, activation="relu")(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(units=10, activation="sigmoid")(x)
        model = Model(inputs=input_layer, outputs=output_layer, name="m3")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        description = None

    elif model_number == 4:
        input_layer = Input(shape=(784,))
        x = Dense(units=512, activation="relu", kernel_regularizer=l2(0.001))(input_layer)
        x = Dense(units=256, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dense(units=128, activation="relu", kernel_regularizer=l2(0.001))(x)
        output_layer = Dense(units=10, activation="softmax", kernel_regularizer=l2(0.001))(x)
        model = Model(inputs=input_layer, outputs=output_layer, name="m4")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        description = None

    elif model_number == 5:
        input_layer = Input(shape=(784,))
        x = Dense(units=512, activation="relu", kernel_regularizer=l2(0.001))(input_layer)
        x = Dense(units=256, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dense(units=128, activation="relu", kernel_regularizer=l2(0.001))(x)
        output_layer = Dense(units=10, activation="sigmoid", kernel_regularizer=l2(0.001))(x)
        model = Model(inputs=input_layer, outputs=output_layer, name="m5")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        description = None

    elif model_number == 6:
        input_layer = Input(shape=(784,))
        x = Dense(units=512, activation="relu", kernel_regularizer=l2(0.01))(input_layer)
        x = Dense(units=256, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dense(units=128, activation="relu", kernel_regularizer=l2(0.01))(x)
        output_layer = Dense(units=10, activation="sigmoid", kernel_regularizer=l2(0.01))(x)
        model = Model(inputs=input_layer, outputs=output_layer, name="m6")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        description = None

    elif model_number == 7:
        input_layer = Input(shape=(784,))
        x = Dense(units=512, activation="relu", kernel_regularizer=l2(0.001))(input_layer)
        x = Dense(units=256, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dense(units=128, activation="relu", kernel_regularizer=l2(0.001))(x)
        output_layer = Dense(units=10, activation="softmax", kernel_regularizer=l2(0.001))(x)
        model = Model(inputs=input_layer, outputs=output_layer, name="m7")
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                      loss=CategoricalCrossentropy(),
                      metrics=["accuracy"])
        description = None

    elif model_number == 8:
        input_layer = Input(shape=(784,))
        x = Dense(units=512, activation="relu", kernel_regularizer=l2(0.001))(input_layer)
        x = Dense(units=256, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dense(units=128, activation="relu", kernel_regularizer=l2(0.001))(x)
        output_layer = Dense(units=10, activation="sigmoid", kernel_regularizer=l2(0.001))(x)
        model = Model(inputs=input_layer, outputs=output_layer, name="m8")
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                      loss=CategoricalCrossentropy(),
                      metrics=["accuracy"])
        description = None

    elif model_number == 9:
        input_layer = Input(shape=(784,))
        x = Dense(units=512, activation="relu", kernel_regularizer=l2(0.01))(input_layer)
        x = Dense(units=256, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dense(units=128, activation="relu", kernel_regularizer=l2(0.01))(x)
        output_layer = Dense(units=10, activation="sigmoid", kernel_regularizer=l2(0.01))(x)
        model = Model(inputs=input_layer, outputs=output_layer, name="m9")
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                      loss=CategoricalCrossentropy(),
                      metrics=["accuracy"])
        description = None

    else:
        raise ValueError(f"❌ Invalid model number: {model_number}")

    # Display model summary (after compilation)
    model.summary()

    return model, description


# Print confirmation message
print("\n✅ model.py successfully executed")