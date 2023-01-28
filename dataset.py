import os
import torch
import librosa
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CFG:
  sr=16000
  n_fft=1024
  hop_length=512
  n_mels=48
  num_output=50
  hidden_size=32

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    
class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=CFG.sr, n_fft=CFG.n_fft, hop_length=CFG.hop_length, n_mels=CFG.n_mels)
        self.db_converter = torchaudio.transforms.AmplitudeToDB()
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000

        log_mel_spec = self.db_converter(self.mel_converter(audio))
        
        return (log_mel_spec, text)
      
    def checkSound(self, itemIndex):
      audio, sample_rate, text, _, _, _ = self.dataset[itemIndex]
      ipd.display(ipd.Audio(audio, rate=CFG.sr))
      print(text)
    
    def checkLogMelSpec(self, itemIndex):
      audio, sample_rate, text, _, _, _ = self.dataset[itemIndex]
      log_mel_spec = self.db_converter(self.mel_converter(audio))
      plot_spectrogram(log_mel_spec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")
      print(text)
        
#dataset = LibriSpeech("test-clean")
#loader = torch.utils.data.DataLoader(dataset, batch_size=16)

# import IPython.display as ipd
# dataset.checkSound(2)
# dataset.checkLogMelSpec(2)

#reference: https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=3CqtR2Fi5-vP