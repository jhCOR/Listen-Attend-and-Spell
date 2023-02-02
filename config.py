import torch

class CFG:
    sr=16000
    n_fft=1024
    hop_length=512
    n_mels=80
    num_output=50
    max_length = 400
    dataset_list = ["dev-clean", "dev-other", "test-clean", "test-other", 
                    "train-clean-100","train-clean-360", "train-other-500"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"