# Import standard libraries
from timeit import default_timer as timer

# Third-party imports
import numpy as np
from keras.api.models import Model
from keras.api.layers import Input, Dense, Dropout
from keras.api.layers import Conv2D, MaxPooling2D, ReLU
from keras.api.layers import Flatten, GlobalAveragePooling2D
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

        input_layer = Input(shape=(28, 28, 1))

        x = Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding="same")(input_layer)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        flattened_layer = Flatten()(x)
        classifier_layer = Dense(units=128, activation="relu")(flattened_layer)
        prediction_layer = Dense(units=10, activation="softmax")(classifier_layer)

        model = Model(inputs=input_layer, outputs=prediction_layer, name="m1")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["categorical_accuracy"])
        description = None

    elif model_number == 2:

        input_layer = Input(shape=(28, 28, 1))

        x = Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding="valid")(input_layer)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="valid")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="valid")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

        flattened_layer = Flatten()(x)
        classifier_layer = Dense(units=128, activation="relu")(flattened_layer)
        prediction_layer = Dense(units=10, activation="softmax")(classifier_layer)

        model = Model(inputs=input_layer, outputs=prediction_layer, name="m2")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["categorical_accuracy"])
        description = None

    elif model_number == 3:

        input_layer = Input(shape=(28, 28, 1))

        x = Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding="valid")(input_layer)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="valid")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)

        flattened_layer = Flatten()(x)
        classifier_layer = Dense(units=128, activation="relu")(flattened_layer)
        prediction_layer = Dense(units=10, activation="softmax")(classifier_layer)

        model = Model(inputs=input_layer, outputs=prediction_layer, name="m3")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["categorical_accuracy"])
        description = None

    elif model_number == 4:

        input_layer = Input(shape=(28, 28, 1))

        x = Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding="same")(input_layer)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        pooling_layer = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(units=10, activation="softmax")(pooling_layer)

        model = Model(inputs=input_layer, outputs=prediction_layer, name="m4")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["categorical_accuracy"])
        description = None

    elif model_number == 5:

        input_layer = Input(shape=(28, 28, 1))

        x = Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding="same")(input_layer)
        x = ReLU()(x)
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding="same")(input_layer)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        pooling_layer = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(units=10, activation="softmax")(pooling_layer)

        model = Model(inputs=input_layer, outputs=prediction_layer, name="m5")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["categorical_accuracy"])
        description = None

    elif model_number == 6:

        input_layer = Input(shape=(28, 28, 1))

        x = Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding="same")(input_layer)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

        pooling_layer = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(units=10, activation="softmax")(pooling_layer)

        model = Model(inputs=input_layer, outputs=prediction_layer, name="m6")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["categorical_accuracy"])
        description = None

    elif model_number == 7:
        from keras.api.applications import MobileNet

        input_shape = (224, 224, 3)
        base_model = MobileNet(input_shape=input_shape, include_top=False, weights="imagenet")
        base_model.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        prediction_layer = Dense(units=10, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=prediction_layer, name="m7_transfer_mobilenet")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["categorical_accuracy"])
        description = "Transfer learning using MobileNet"

        # Adjust summary and inference time
        model.summary()
        start_time = timer()
        _ = model(np.random.rand(1, 224, 224, 3))
        elapsed_time = timer() - start_time
        print("\n🔹 Inference Time:\n")
        print(f"{elapsed_time:.6f} seconds (m7)")

        return model, description

    else:
        raise ValueError(f"❌ Invalid model number: {model_number}")

    # Display model summary (after compilation)
    model.summary()

    # Measure inference time
    start_time = timer()
    _ = model(np.random.rand(32, 28, 28, 1))
    elapsed_time = timer() - start_time
    print("\n🔹 Inference Time:\n")
    print(f"{elapsed_time:.6f} seconds (m{model_number})")

    return model, description


# Print confirmation message
print("\n✅ model.py successfully executed")