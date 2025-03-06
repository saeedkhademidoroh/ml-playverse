import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_model_history(history):
    """Plots training history."""
    print("\n🎯 Visualizing Training History 🎯\n")
    pd.DataFrame(history.history).plot(figsize=(10, 5))
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()