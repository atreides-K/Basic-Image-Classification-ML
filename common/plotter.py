# --- Example: Basic Plotting (Loading from .npz) ---
import matplotlib.pyplot as plt

# Load the training_plot back (if plotting in a separate step/script)
# print(f"Loading training_plot from {history_save_path}...")
# loaded_history_npz = np.load(history_save_path)
# history_loaded = {key: loaded_history_npz[key] for key in loaded_history_npz}
# print("History loaded.")
# Use 'training_plot' directly if plotting immediately after training

epochs_range = range(1, config["epochs"] + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Use the training_plot dictionary directly if plotting right after training
plt.plot(epochs_range, training_plot['train_loss'], label='Train Loss')
plt.plot(epochs_range, training_plot['test_loss'], label='Test Loss')
# Or use the loaded data: plt.plot(epochs_range, history_loaded['train_loss'], ...)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title(f'ResNet-{config["depth"]} Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_plot['train_acc'], label='Train Accuracy')
plt.plot(epochs_range, training_plot['test_acc'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'ResNet-{config["depth"]} Accuracy Curve')

plt.tight_layout()
plot_save_path = config['save_name'].replace(".pth", "_curves.png")
print(f"Saving plot to {plot_save_path}...")
plt.savefig(plot_save_path)
plt.close() # Close the plot figure
print("Plot saved.")
