import tensorflow as tf
import datetime

# Set the log directory
log_dir = "logs/fit/"

# Print instructions for running TensorBoard
print("To view TensorBoard, run the following command in your terminal:")
print(f"tensorboard --logdir {log_dir}")
print("\nThen open a web browser and go to: http://localhost:6006")

# If you're running this in a Jupyter notebook, you can use the following to display TensorBoard inline:
# %load_ext tensorboard
# %tensorboard --logdir {log_dir}
