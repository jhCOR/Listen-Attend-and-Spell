from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from torchaudio import transforms, functional
import torch.nn.functional as F
import random
# from specaugment.spec_augment_pytorch import spec_augment

class SpeechDataset(Dataset):

    def __init__(self, char2id, split=CFG.dataset_list[4], max_len=0, specaugment=False, pkwargs=None):
        
        super(SpeechDataset, self).__init__()
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.char2id = char2id
        self.max_len = max_len
        self.specaugment = specaugment
        self.pkwargs = pkwargs
        # if self.specaugment:
        #     self.origin_data = list(self.pair_data)
        #     self.aug_ids = [id for id in range(len(self.pair_data), len(self.pair_data)*2)]
        #     self.pair_data.extend(self.pair_data)
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=CFG.sr, n_fft=CFG.n_fft, hop_length=CFG.hop_length, n_mels=CFG.n_mels)
        self.db_converter = torchaudio.transforms.AmplitudeToDB()
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio, sample_rate, label, _, _, _ = self.dataset[idx]

        log_mel_spec = self.db_converter(self.mel_converter(audio))
        # print("audio.unsqueeze(0):", audio.unsqueeze(0).shape)
        # x = torchaudio.compliance.kaldi.fbank(audio, num_mel_bins=80).t().unsqueeze(0)
        # print("x shape:", x.shape)
        # (m, 80 + use_energy)

        # if self.specaugment:
        #     if idx in self.aug_ids:
        #         x = spec_augment(x, time_warping_para=40, frequency_masking_para=13,
        #                         time_masking_para=30, frequency_mask_num=2, time_mask_num=2)
        
        x = log_mel_spec
        x = x[0,:,:].squeeze(1).t()
        if self.max_len:
            x = np.pad(x, ((0, 0), (0, self.max_len - x.shape[1])), "constant")
            
        y = []
        y.append(SOS_TOKEN)
        for char in label:
            try:
                y.append(self.char2id[char])
            except:
                y.append(self.char2id['<unk>'])
        y.append(EOS_TOKEN)
        y = np.array(y)
        return (x, y)
    
    def shuffle(self):
        random.shuffle(self.dataset)
        # if self.specaugment:
        #     self.aug_ids = [id for id in range(len(self.pair_data), len(self.pair_data)*2)]
        #     self.pair_data.extend(self.pair_data)
            
    def trim(self, sig, hop_size=64, threshhold=0.002):
        head = None
        tail = None
        #threshhold = ((sig.max())/2)*threshhold
        sig_len = len(sig)
        for i in range(int(sig_len/hop_size)):
            pre = sig[i*hop_size:(i+1)*hop_size].abs().sum().item()
            post = sig[(i+1)*hop_size:(i+2)*hop_size].abs().sum().item()
            grad = abs((post-pre)/hop_size)
            if grad>threshhold:
                head = (i+1)*hop_size
                break

        for i in range(int(sig_len/hop_size)):
            pre = sig[sig_len-(i+1)*hop_size:sig_len-i*hop_size].abs().sum().item()
            post = sig[sig_len-(i+2)*hop_size:sig_len-(i+1)*hop_size].abs().sum().item()
            grad = abs((post-pre)/hop_size)
            if grad>threshhold:
                tail = sig_len-(i+1)*hop_size
                break
        #print(head, tail)
        return sig[head:tail]

def _collate_fn(batch):
    """ functions that pad to the maximum sequence length """
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(0)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    return seqs, targets