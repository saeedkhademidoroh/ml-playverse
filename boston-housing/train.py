from config import BATCH_SIZE, EPOCHS
from keras.api.callbacks import EarlyStopping

def train_model(model, train_data, train_labels, test_data, test_labels):
    """Trains the model and returns the training history."""
    print("\n🎯 Training Model 🎯\n")

    early_stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

    history = model.fit(
        x=train_data, y=train_labels,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(test_data, test_labels),
        callbacks=[early_stop]
    )

    return history
