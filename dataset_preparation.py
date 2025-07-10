import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

folder_path = r"/home/arobin/Documents/audio_u_net/musdb18hq"
train_path = r"/mnt/storage2/arobin/wave_u_net/test"
sampling_Rate = 44100

# Define file extensions for order
file_extension = ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]


def find_len(folder_path):
    l = 0
    ## read a random file from each folder
    for folder_name in os.listdir(train_path):
        w, f = torchaudio.load(os.path.join(train_path, folder_name, "mixture.wav"))
        l = max(l, w.shape[1])

    return l


# Set max audio length (precomputed to avoid crash)
max_len = find_len(train_path)
print(f"maximum length is {max_len}")


# Function to pad audio efficiently
def pad_audio(waveform, target_length):
    if waveform.shape[1] < target_length:
        return torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
    return waveform[:, :target_length]  # Truncate if longer


# **Define Dataset Class**
class AudioDataset(Dataset):
    def __init__(self, train_path, file_extension):
        self.train_path = train_path
        self.file_extension = file_extension
        self.folders = os.listdir(train_path)  # List all folders
        self.max_len = find_len(train_path)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        folder_full_path = os.path.join(self.train_path, folder)
        temp = [None] * len(self.file_extension)  # Initialize ordered storage
        input_audio = []
        all_folder = os.listdir(folder_full_path)
        ## take only 30% of the data 
        train_folder = all_folder[:int(len(all_folder)*0.3)]
        for file in train_folder:
            file_full_path = os.path.join(folder_full_path, file)
            wavefile, sampling = torchaudio.load(file_full_path)

            # Ensure mono audio
            if wavefile.shape[0] == 2:
                wavefile = wavefile.mean(dim=0, keepdim=True)

            assert sampling == sampling_Rate

            wavefile = pad_audio(wavefile, self.max_len)

            if file in self.file_extension:
                temp[self.file_extension.index(file)] = wavefile
            else:
                input_audio.append(wavefile)

        # Convert None to zero tensors for missing audio files
        target_audio = [
            t if t is not None else torch.zeros((1, self.max_len)) for t in temp
        ]

        inp = torch.stack(input_audio)
        tar = torch.stack(target_audio)
        return inp.squeeze(1), tar.squeeze(1)


if __name__ == "__main__":
    # **Create Dataset and DataLoader**
    batch_size = 30  # Adjust batch size based on RAM availability
    dataset = AudioDataset(train_path, file_extension)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # **Iterate Over DataLoader**
    print("Loading data in batches...")
    for batch_idx, (input_wavefile, target_wavefile) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f" - Input Tensor Shape: {input_wavefile.shape}")
        print(f" - Target Tensor Shape: {target_wavefile.shape}")

    print("Audio loading completed in batches!")
