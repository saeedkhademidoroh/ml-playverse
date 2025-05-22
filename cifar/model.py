# Import third-party libraries
from keras.api.models import Model
from keras.api.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    BatchNormalization, Activation, GlobalAveragePooling2D,
    DepthwiseConv2D, Add, Dropout
)
from keras.api.optimizers import Adam, SGD
from keras.api.losses import SparseCategoricalCrossentropy
from keras.api.regularizers import l2


# Function to build a model
def build_model(model_number: int, config) -> Model:
    """
    Function to build and compile a model based on the given model_number.

    Supports several variants of VGG-style CNNs with optional BatchNorm,
    GlobalAveragePooling, separable convolutions, residual connections,
    and configurable regularization (L2 and Dropout).

    Args:
        model_number (int): Identifier for architecture variant

    Returns:
        Model: A compiled Keras model instance
    """

    # Print header for function execution
    print("\n🎯  build_model\n")

    # Extract optimizer configuration from config
    optimizer_config = config.OPTIMIZER
    optimizer_type = optimizer_config.get("type", "adam").lower()
    learning_rate = optimizer_config.get("learning_rate", 0.001)
    momentum = optimizer_config.get("momentum", 0.0)

    # Initialize optimizer according to config
    if optimizer_type == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"❌  ValueError from model.py in build_model():\noptimizer_type={optimizer_type}\n")

    # Define L2 regularizer if enabled
    regularizer = l2(config.L2_MODE["lambda"]) if config.L2_MODE.get("enabled", False) else None

    # Dropout utility for optional injection
    def maybe_dropout(x):
        return Dropout(config.DROPOUT_MODE["rate"])(x) if config.DROPOUT_MODE.get("enabled", False) else x

    input_layer = Input(shape=(32, 32, 3))

    # Model 0
    if model_number == 0:
        # Block 1: Conv(32) x2 + MaxPool
        x = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizer)(input_layer)
        x = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizer)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(64) x2 + MaxPool
        x = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizer)(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizer)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Flatten + Dense(128)
        x = Flatten()(x)
        x = Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
        x = maybe_dropout(x)
        prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)

    # Model 1
    elif model_number == 1:
        # Block 1: Conv(32) x2 + BN + ReLU + MaxPool
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(64) x2 + BN + ReLU + MaxPool
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Flatten + Dense(128)
        x = Flatten()(x)
        x = Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
        x = maybe_dropout(x)
        prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)

    # Model 2
    elif model_number == 2:
        # Block 1: Conv(16) x2 + BN + ReLU + MaxPool
        x = Conv2D(16, (3, 3), padding="same", kernel_regularizer=regularizer)(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(16, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(32) x2 + BN + ReLU + MaxPool
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GlobalAveragePooling2D()(x)
        x = maybe_dropout(x)
        prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)

    # Model 3
    elif model_number == 3:
        # Block 1: Conv(32) x2 + BN + ReLU + MaxPool
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(64) x2 + BN + ReLU + MaxPool
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GlobalAveragePooling2D()(x)
        x = maybe_dropout(x)
        prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)

    # Model 4
    elif model_number == 4:
        # Block 1: (Depthwise + Pointwise Conv) x2 + BN + ReLU + MaxPool
        x = DepthwiseConv2D((3, 3), padding="same")(input_layer)
        x = Conv2D(32, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = DepthwiseConv2D((3, 3), padding="same")(x)
        x = Conv2D(32, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: (Depthwise + Pointwise Conv) x2 + BN + ReLU + MaxPool
        x = DepthwiseConv2D((3, 3), padding="same")(x)
        x = Conv2D(64, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = DepthwiseConv2D((3, 3), padding="same")(x)
        x = Conv2D(64, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GlobalAveragePooling2D()(x)
        x = maybe_dropout(x)
        prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)

    # Model 5
    elif model_number == 5:
        # Block 1: Conv(32) x2 + BN + Add shortcut + ReLU + MaxPool
        shortcut = Conv2D(32, (1, 1), padding="same", kernel_regularizer=regularizer)(input_layer)
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(64) x2 + BN + Add shortcut + ReLU + MaxPool
        shortcut = Conv2D(64, (1, 1), padding="same", kernel_regularizer=regularizer)(x)
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GlobalAveragePooling2D()(x)
        x = maybe_dropout(x)
        prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)

    # Model 6
    elif model_number == 6:
        # Initial Conv(32)
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Residual Block 1a: Conv(32) x2 + Add + ReLU
        shortcut = x
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)

        # Residual Block 1b: same as 1a
        shortcut = x
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)

        # Residual Block 2a: downsample + increase filters to 64
        shortcut = Conv2D(64, (1, 1), strides=2, padding="same", kernel_regularizer=regularizer)(x)
        x = Conv2D(64, (3, 3), strides=2, padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)

        # Residual Block 2b: Conv(64) x2 + Add + ReLU
        shortcut = x
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)

        # Residual Block 3a: downsample + increase filters to 128
        shortcut = Conv2D(128, (1, 1), strides=2, padding="same", kernel_regularizer=regularizer)(x)
        x = Conv2D(128, (3, 3), strides=2, padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)

        # Residual Block 3b: Conv(128) x2 + Add + ReLU
        shortcut = x
        x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)

        x = GlobalAveragePooling2D()(x)
        x = maybe_dropout(x)
        prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)

    # Model 7
    elif model_number == 7:
        # Initial Conv(20)
        x = Conv2D(20, (3, 3), padding="same", kernel_regularizer=regularizer)(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Stage 1: 3 blocks @ 20 filters
        for _ in range(3):
            x = res_block(x, 20, regularizer)

        # Stage 2: 3 blocks @ 40 filters, first with downsampling
        x = res_block(x, 40, regularizer, downsample=True)
        for _ in range(2):
            x = res_block(x, 40, regularizer)

        # Stage 3: 3 blocks @ 80 filters, first with downsampling
        x = res_block(x, 80, regularizer, downsample=True)
        for _ in range(2):
            x = res_block(x, 80, regularizer)

        # Final layers
        x = GlobalAveragePooling2D()(x)
        x = maybe_dropout(x)
        prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)

    # Model 8
    elif model_number == 8:
        # Initial Conv(16)
        x = Conv2D(16, (3, 3), padding="same", kernel_regularizer=regularizer)(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Stage 1: 3 blocks @ 16 filters
        for _ in range(3):
            x = res_block(x, 16, regularizer)

        # Stage 2: 3 blocks @ 32 filters, first with downsampling
        x = res_block(x, 32, regularizer, downsample=True)
        for _ in range(2):
            x = res_block(x, 32, regularizer)

        # Stage 3: 3 blocks @ 64 filters, first with downsampling
        x = res_block(x, 64, regularizer, downsample=True)
        for _ in range(2):
            x = res_block(x, 64, regularizer)

        # Final layers
        x = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)


    else:
        raise ValueError(f"❌  ValueError from model.py at build_model():\nmodel_number={model_number}\n")

    # Compile model with selected optimizer and loss/metrics
    model = Model(inputs=input_layer, outputs=prediction_layer)
    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])

    # Print model architecture summary
    model.summary()

    return model  # Return compiled Keras model instance


def res_block(x, filters, regularizer, downsample=False):
    stride = 2 if downsample else 1
    shortcut = x
    x = Conv2D(filters, (3, 3), strides=stride, padding="same", kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same", kernel_regularizer=regularizer)(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


# Print module successfully executed
print("\n✅  model.py successfully executed")