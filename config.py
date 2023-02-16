import torch

class CFG:
    sr=16000
    n_fft=1024
    hop_length=512
    n_mels=100
    num_output=50
    max_length = 500
    
    batch_size = 4 #16
    worker = 2     #16
    num_channels = 100 #80
    
    dataset_list = ["dev-clean", "dev-other", "test-clean", "test-other", 
                    "train-clean-100","train-clean-360", "train-other-500"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

char_list = ['<pad>', '<sos>', '<eos>', '<unk>', " ", "A", "B", "C", "D", "E", "F", "G","H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'"]
char2id = {v:k for k,v in enumerate(char_list)}
id2char = {k:v for k,v in enumerate(char_list)}

PAD_TOKEN = int(char2id['<pad>'])
SOS_TOKEN = int(char2id['<sos>'])
EOS_TOKEN = int(char2id['<eos>'])
UNK_TOKEN = int(char2id['<unk>'])