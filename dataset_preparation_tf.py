import os
import tensorflow as tf
import numpy as np
import soundfile as sf

folder_path = "/home/arobin/Documents/audio_u_net/musdb18hq"
train_path = "/mnt/storage2/arobin/wave_u_net/test"
sampling_rate = 44100

# Define file extensions for order
file_extensions = ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]


def find_max_len(folder_path):
    max_length = 0
    for folder_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, folder_name, "mixture.wav")
        if os.path.exists(file_path):
            audio, sr = sf.read(file_path)
            max_length = max(max_length, audio.shape[0])
    return max_length


# Set max audio length (precomputed to avoid crash)
max_len = find_max_len(train_path)
print(f"Maximum length is {max_len}")


def pad_audio(audio, target_length):
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), mode="constant")
    return audio[:target_length]  # Truncate if longer


class AudioDataset(tf.keras.utils.Sequence):
    def __init__(self, train_path, file_extensions, batch_size=30):
        self.train_path = train_path
        self.file_extensions = file_extensions
        self.folders = os.listdir(train_path)
        ## take only 30% of the data 
        self.folders = self.folders[:int(len(self.folders)*0.3)]
        self.max_len = 8192 #int(np.sqrt(find_max_len(train_path)))**2
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.folders) / self.batch_size))

    def __getitem__(self, index):
        batch_folders = self.folders[index * self.batch_size:(index + 1) * self.batch_size]
        input_audio_list, target_audio_list = [], []

        for folder in batch_folders:
            folder_full_path = os.path.join(self.train_path, folder)
            temp = [None] * len(self.file_extensions)
            input_audio = []

            for file in os.listdir(folder_full_path):
                file_full_path = os.path.join(folder_full_path, file)
                wavefile, sr = sf.read(file_full_path)

                # Ensure mono audio
                if len(wavefile.shape) > 1:
                    wavefile = np.mean(wavefile, axis=1)
                
                assert sr == sampling_rate

                wavefile = pad_audio(wavefile, self.max_len)

                if file in self.file_extensions:
                    temp[self.file_extensions.index(file)] = wavefile
                else:
                    input_audio.append(wavefile)

            # Convert None to zero tensors for missing audio files
            target_audio = [t if t is not None else np.zeros((self.max_len,)) for t in temp]
            input_audio_list.append(np.stack(input_audio, axis=0))
            target_audio_list.append(np.stack(target_audio, axis=0))
        x = tf.transpose(np.array(input_audio_list),perm=[0, 2, 1])
        y = np.array(target_audio_list)
        #print(f' shape of x  {x.shape} ,  and y {y.shape}')
        return x, y

if __name__ == "__main__":
    batch_size = 30
    dataset = AudioDataset(train_path, file_extensions, batch_size=batch_size)

    print("Loading data in batches...")
    for batch_idx in range(len(dataset)):
        input_wavefile, target_wavefile = dataset[batch_idx]
        print(f'len of the dataset {len(dataset)}')
        print(f"Batch {batch_idx + 1}:")
        print(f" - Input Tensor Shape: {input_wavefile.shape}")
        print(f" - Target Tensor Shape: {target_wavefile.shape}")

    print("Audio loading completed in batches!")
