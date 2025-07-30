import torchaudio
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json, read_txt


class ASVspoof(BaseDataset):
    def __init__(
        self, name = "train", trial_file = "ASVspoof2019.LA.cm.train.trn.txt", audio = None, *args, **kwargs
    ):
        """
        Args:
            trial_file (str): path to .trn file
            audio (Path | str): root path to audio files (.flac)
            name (str): partition name
        """

        index_path = ROOT_PATH / "data" / "ASVspoof2019_LA" / name / "index.json"
        self.audio = ROOT_PATH / "data" / "ASVspoof2019_LA" / f"{name}/flac"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(trial_file, index_path)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, trial_file, index_path):
        index = []
        data_path = ROOT_PATH / "data" / "ASVspoof2019_LA" / trial_file
        data = read_txt(str(data_path))

        for i in tqdm(data):
            splited = i.strip().split()
            id, label = splited[0], int(splited[-1]=="bonafide")
            index.append({"path": str(self.audio / f"{id}.flac"), "label": label})

        # write index to disk
        write_json(index, index_path)

        return index
        
    def load_object(self, path):
        waveform, sample_rate = torchaudio.load(path)
        stft = torch.stft(
            waveform,
            pad=0,
            window=torch.hann_window(400),
            n_fft=1732,
            hop_length=160,
            win_length=400,
            power=2.0,
        )
        magnitude = stft.sqrt().mean(dim=0, keepdim=True)

        if magnitude.shape[3] < 600:
            pad_time = 600 - magnitude.shape[3]
            magnitude = torch.nn.functional.pad(magnitude, (0, pad_time, 0, 0))
        elif magnitude.shape[3] > 600:
            magnitude = magnitude[:, :, :600]
        return magnitude
