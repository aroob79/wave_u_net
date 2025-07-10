import tensorflow as tf
import numpy as np
import os
import time
import soundfile as sf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard 
from model_tf import UNet1D 
from dataset_preparation_tf import AudioDataset 

# Training setup
folder_path = "/mnt/storage2/arobin/wave_u_net/test"
batch_size = 5
learning_rate = 1e-3
# Define file extensions for order
file_extensions = ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]
dataset = AudioDataset(folder_path,file_extensions, batch_size=batch_size)
model = UNet1D(in_channels=1, out_channels=4)

# Loss Function
@tf.function
def sdr_loss(y_true, y_pred, eps=1e-8):
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
    true_norm = tf.reduce_sum(y_true ** 2, axis=-1, keepdims=True) + eps
    pred_norm = tf.reduce_sum(y_pred ** 2, axis=-1, keepdims=True) + eps
    return -tf.reduce_mean(dot_product / tf.sqrt(true_norm * pred_norm))

@tf.function
def combined_loss(y_true, y_pred, alpha=0.8):
    return alpha * tf.keras.losses.MeanAbsoluteError()(y_true, y_pred) + (1 - alpha) * sdr_loss(y_true, y_pred)

# Optimizer and Learning Rate Scheduler
lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=1000, decay_rate=0.7)
optimizer = Adam(learning_rate=lr_schedule)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Callbacks
checkpoint_callback = ModelCheckpoint("model_checkpoint.h5", save_best_only=True)
tensorboard_callback = TensorBoard(log_dir="logs")

# Train Model
epochs = 20

with tf.device('/GPU:0'):
    # Compile Model
    model.compile(optimizer=optimizer, loss=combined_loss)
    model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback, tensorboard_callback])
