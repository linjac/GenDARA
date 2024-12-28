import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule
import scipy.signal as signal
from typing import Tuple, List, Dict

class SdeDataset(Dataset):
    def __init__(self, 
                 path_audios, 
                 df, 
                 path_speech=None, 
                 sr=32000, 
                 duration=10):
        if not os.path.exists(path_audios):
            raise ValueError(f"The path {path_audios} does not exist.")
        self.path_audios = path_audios
        if path_speech is not None:
            self.path_speech = path_speech
        else:
            self.path_speech = os.path.join(path_audios, "speech")
        self.list_all_speech_files = [f for f in os.listdir(self.path_speech) if f.endswith('.wav')]

        self.sr = sr
        self.duration = duration # duration of the reverberant signal in seconds
        self.meta_df = df
        if 'convolved_filename' in self.meta_df.columns:
            self.convolved = {} 
        self.filters = {}
        self.speech = {}
    
    def __len__(self):
        return len(self.meta_df)
        
    def __getitem__(self, index):
        filter_info = self.meta_df.iloc[index]
        
        # Get filter filename and speech filename
        filter_filename = filter_info['filename']
        if 'speech_filename' in filter_info: # set speech_filename to the speech file specified for this IR filter. 
            speech_filename = filter_info['speech_filename']
        elif 'convolved_filename' not in filter_info:  # Otherwise, select random file
            speech_filename = self.list_all_speech_files[np.random.randint(0, len(self.list_all_speech_files))]
        else:  # convoled file does not specify a speech file
            speech_filename = 'None specified'
            
        # Load convolved file if it exists, otherwise convolve the filter with the speech
        if 'convolved_filename' not in filter_info: # no existing convolved file, convolve the filter with the speech
            filter = self._get_filter(filter_filename)
            speech = self._get_speech(speech_filename)
            sound = self._apply_filter(filter, speech, self.duration) # convolve
            sound = sound.squeeze()
        else:
            sound = self._get_convolved(filter_info['convolved_filename']) # load the convolved file
        
        # Set the label
        distance = filter_info['dist_gt']
        
        return {
            "audio": torch.tensor(sound).float(), 
            "label": torch.tensor(distance).float(),
            "id": filter_filename,
            "speech_id": speech_filename
            }
    
    def _get_filter(self, filter_filename):
        filter = self.filters.get(filter_filename, None)  # Look for the filter in self.filters
        if filter is None:
            path = os.path.join(self.path_audios, filter_filename)
            filter, _ = lb.load(path, sr = self.sr, mono = True, res_type = "kaiser_fast")
            self.filters[filter_filename] = filter
        return filter
    
    def _get_speech(self, speech_filename):
        speech = self.speech.get(speech_filename, None)  # Look for the speech in self.speech
        if speech is None:
            path = os.path.join(self.path_speech, speech_filename)
            speech, _ = lb.load(path, sr = self.sr, mono = True, res_type = "kaiser_fast")
            self.speech[speech_filename] = speech
        return speech
    
    def _get_convolved(self, convolved_filename):
        sound = self.convolved.get(convolved_filename, None)  # Look for the convolved audio in self.convolved
        if sound is None:
            path = os.path.join(self.path_audios, convolved_filename)
            sound, _ = lb.load(path, sr = self.sr, mono = True, res_type = "kaiser_fast")
            self.convolved[convolved_filename] = sound
        return sound

    def _apply_filter(self, filter, speech, duration):
        '''apply the filter to the speech signal'''
        repeat_til_duration = lambda y: np.tile(y, (duration*self.sr)//len(y) + 1) if len(y) < duration*self.sr else y[:duration*self.sr]
        speech_data = repeat_til_duration(speech) # Make sure the speech is at least duration seconds long
        
        # Convolve the signals
        convolved = signal.convolve(speech_data, filter)[:duration*self.sr]
        # Normalize the output
        if np.max(np.abs(convolved)) > 0.99:
            convolved = convolved / np.max(np.abs(convolved))
        return convolved
    
    def get_mean_distances(self):
        distance = 0
        for i in range(len(self.meta_df)):
            returned_values = self.__getitem__(i)
            distance += returned_values['label']
        return distance / len(self.meta_df)
    
    def get_distribution(self):
        distances = []
        for i in range(len(self.list_all_files)):
            returned_values = self.__getitem__(i)
            distances.append(returned_values['label'].numpy())
        distances = np.array(distances)
        plt.figure()
        plt.hist(distances, edgecolor = 'k', alpha = 0.65)
        plt.axvline(distances.mean(), color='r', linestyle='dashed', linewidth=1)
        _, max_ylim = plt.ylim()
        plt.text(distances.mean()*1.05, max_ylim*0.9, 'Mean: {:.2f} m'.format(distances.mean()))
        plt.grid(alpha = 0.2)
        plt.title("Speaker distance distribution")
        plt.xlabel("Distance [m]")
        plt.ylabel("Count")
        plt.savefig("Speaker_distantce_distribution.pdf", transparent = True)
        plt.show()
        
        
class SdeDataModule(LightningDataModule):
    
    def __init__(self, path_dataset, path_speech: Dict, dfs: Dict, batch_size, sr: int, duration: float, dataloader_config = None):
        super().__init__()
        self.path_dataset = path_dataset
        self.train_df = dfs["train"]
        self.val_df = dfs["val"]
        self.test_df = dfs["test"]
        self.train_path_speech = path_speech["train"]
        self.val_path_speech = path_speech["val"]
        self.test_path_speech = path_speech["test"]
        
        self.sr = sr
        self.duration = duration
        self.batch_size = batch_size
        
        self.dataloader_config = dataloader_config
        
    def prepare_data(self):
        '''
            Nothing to do
        '''
        pass
        
    def setup(self, stage = None):
        '''
        Nothing to do
        '''
        pass

    def train_dataloader(self):
        return DataLoader(SdeDataset(self.path_dataset, self.train_df, path_speech=self.train_path_speech, sr=self.sr, duration=self.duration),
                          batch_size = self.batch_size, shuffle = True, drop_last = True,
                          num_workers=self.dataloader_config["num_workers"], pin_memory=self.dataloader_config["pin_memory"]
                          )
    
    def val_dataloader(self):
        return DataLoader(SdeDataset(self.path_dataset, self.val_df, path_speech=self.val_path_speech, sr=self.sr, duration=self.duration) , 
                          batch_size = self.batch_size, shuffle = False, drop_last = False,
                          num_workers=self.dataloader_config["num_workers"], pin_memory=self.dataloader_config["pin_memory"])
    
    def test_dataloader(self):
        return DataLoader(SdeDataset(self.path_dataset, self.test_df, path_speech=self.test_path_speech, sr=self.sr, duration=self.duration), 
                          batch_size = self.batch_size, shuffle = False, drop_last = False)
