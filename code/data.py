# -*- coding: utf-8 -*-
import os
import glob
import copy
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.transforms import LogTransform, NormalizeTensor, NoiseGaussian
from utils.plot import plot_batch, plot_batch_detailed
from ddsp.analysis import Loudness, FundamentalFrequency, MultiscaleFFT
import librosa

"""
###################

Raw audio and features dataset definition

###################
"""
class AudioFeaturesDataset(Dataset):
    """
    Create a dataseet of mixed audio, fft and features. Each of the original 
    audio tracks will be processed by first slicing it into blocks X sequence
    elements. Each element is the used as input to obtain
    - Multi-scale FFT information
    - Fundamental frequency
    - Loudness
    
    Args:
        datadir (str): Percentage to be put to zero. default: .2
    """
    
    def __init__(self, datadir, args, transform=None, splits=[.8, .1, .1], shuffle_files=True, train='train'):
        self.args = args
        # Metadata and raw
        self.data_files = []
        # Spectral transforms
        self.features_files = []
        # Construct set of extractors
        self.construct_extractors(args)
        # Construct the FFT extractor
        self.multi_fft = MultiscaleFFT(args.scales)
        # Retrieve list of files
        tmp_files = sorted(glob.glob(datadir + '/raw/*.wav'))
        self.data_files.extend(tmp_files)
        print(self.data_files)
        if (not os.path.exists(datadir + '/data') or len(glob.glob(datadir + '/data/*.npz')) == 0):
            os.makedirs(datadir + '/data')
            self.preprocess_dataset(datadir)
        feat_files = sorted(glob.glob(datadir + '/data/*.npz'))
        self.features_files.extend(feat_files)
        # Analyze dataset
        self.analyze_dataset()
        # Create splits
        self.create_splits(splits, shuffle_files)
        # Compute mean and std of dataset
        self.compute_normalization()
        # Now we can create the normalization / augmentation transform
        self.transform = transform
        # Otherwise create a basic normalization / augmentation transform
        if (transform is None):
            tr = []
            tr.append(LogTransform(clip=1e-3))
            tr.append(NormalizeTensor(self.mean, self.var))
            tr.append(transforms.RandomApply([NoiseGaussian(factor=1e-2)], p=0.333))
            #tr.append(transforms.RandomApply([OutliersZeroRandom(factor=.1)], p=0.333))
            self.transform = transforms.Compose(tr)
    
    def construct_extractors(self, args):
        self.extractors = {}
        self.extractors['f0'] = FundamentalFrequency(args.sr, args.block_size, args.sequence_size)
        self.extractors['loudness'] = Loudness(args.block_size, args.kernel_size)
    
    def preprocess_dataset(self, datadir):
        cur_id = 0
        for cur_file in self.data_files:
            # Keep the current file name
            f_name, _ = os.path.splitext(os.path.basename(cur_file))
            # Import audio
            y, sr = librosa.core.load(cur_file) 
            # Compute all sequences
            mod = self.args.block_size * self.args.sequence_size
            # Reshape into batch x seq
            y = y[:mod*(len(y)//mod)].reshape(-1,mod)
            # Compute the full multi-scales FFT
            cur_fft = self.multi_fft(torch.from_numpy(y))
            # Compute each batch feature
            for i in range(y.shape[0]):
                features = {}
                features['name'] = f_name
                features['audio'] = y[i]
                features['fft'] = cur_fft[i]
                # Compute features in dataset
                for k, v in self.extractors.items():
                    features[k] = v(y[i])
                # Save to numpy compressed format
                np.savez(datadir + '/data/seq_' + str(cur_id) + '.npz', features)
                cur_id += 1    
    
    def switch_set(self, name):
        if (name == 'test'):
            self.data_files = self.test_files[0]
            self.features_files = self.test_files[1]
        if (name == 'valid'):
            self.data_files = self.valid_files[0]
            self.features_files = self.valid_files[1]
        tr = []
        tr.append(LogTransform(clip=1e-3))
        tr.append(NormalizeTensor(self.mean, self.var))
        self.transform = transforms.Compose(tr)
        self.test_files = None
        self.valid_files = None
        return self
            
    def compute_normalization(self):
        self.mean = 0
        self.var = 1
        return
        tr = LogTransform(clip=1e-3)
        # Parse dataset to compute mean and norm
        for n in range(len(self.spectral_files)):
            data = np.load(self.spectral_files[n], allow_pickle=True)
            data = torch.from_numpy(data).float()
            data = tr(data)
            # Current file stats
            b_mean = data.mean()
            b_var = (data - self.mean)
            # Running mean and var
            self.mean = self.mean + ((b_mean - self.mean) / (n + 1))
            self.var = self.var + ((data - self.mean) * b_var).mean()
        self.mean = float(self.mean)
        self.var = float(np.sqrt(self.var / len(self.spectral_files)))
    
    def analyze_dataset(self):
        # Fill some properties based on the first file
        loaded = np.load(self.features_files[0], allow_pickle=True)
        self.input_size = data.shape
        # Metadata indicators (character = 0, features = 1, categories = 2)
        self.metadata = torch.zeros(len(self.data_files), 1)
        # Check parameters
        params_check = []
        # Parse dataset to check parameters
        for n in range(len(self.data_files)):
            file = np.load(self.data_files[n], allow_pickle=True) 
            loaded = file['param'].item()
            params_check.append(torch.Tensor([loaded[p] for p in self.param_names]).unsqueeze(0))
            meta_keys = file['chars']
            self.metadata[n] = (np.sum(meta_keys[:, 2]) == meta_keys.shape[0]) * 1.0
        full_params = torch.cat(params_check, dim=0)
        self.params_std = torch.std(full_params, dim=0)
        self.params_mean = torch.mean(full_params, dim=0)
        self.params_idx = self.params_std.nonzero()
        self.final_params = self.param_names#[self.param_names[i] for i in self.params_idx]
        self.final_std = self.params_std#[self.params_std[i] for i in self.params_idx]
        self.output_size = len(self.final_params)
            
    def create_splits(self, splits, shuffle_files):
            nb_files = len(self.data_files)
            if (shuffle_files):
                idx = np.random.permutation(nb_files).astype('int')
                self.data_files = [self.data_files[i] for i in idx]
                self.features_files = [self.features_files[i] for i in idx]
            idx = np.linspace(0, nb_files-1, nb_files).astype('int')
            train_idx = idx[:int(splits[0]*nb_files)]
            valid_idx = idx[int(splits[0]*nb_files):int((splits[0]+splits[1])*nb_files)]
            test_idx = idx[int((splits[0]+splits[1])*nb_files):]
            # Validation split
            self.valid_files = (
                    [self.data_files[i] for i in valid_idx], 
                    [self.features_files[i] for i in valid_idx])
            # Test split
            self.test_files = (
                    [self.data_files[i] for i in test_idx], 
                    [self.features_files[i] for i in test_idx])
            self.data_files = [self.data_files[i] for i in train_idx]
            self.features_files = [self.features_files[i] for i in train_idx]

    def __getitem__(self, idx):
        loaded = np.load(self.features_files[idx], allow_pickle=True)
        audio = torch.from_numpy(loaded['audio'])
        loudness = torch.from_numpy(loaded['loudness'])
        fft = torch.from_numpy(loaded['fft'])
        f0 = torch.from_numpy(loaded['f0'])
        # Eveentual pre-processing to apply
        #data = self.transform(torch.from_numpy(data).float())
        return fft, f0, loudness, audio

    def __len__(self):
        return len(self.data_files)
    
"""
###################

Load any given dataset and return DataLoaders

###################
"""
def load_dataset(args, **kwargs):
    if (args.dataset in ['violin', 'violin_simple']):
        dset_train = AudioFeaturesDataset(args.path + '/' + args.dataset, args, **kwargs)
        dset_valid = copy.deepcopy(dset_train).switch_set('valid')
        dset_test = copy.deepcopy(dset_train).switch_set('test')
        dset_train = dset_train.switch_set('train')
    else:
        raise Exception('Wrong name of the dataset!')
    args.input_size = dset_train.input_size
    args.output_size = dset_train.output_size
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.nbworkers, pin_memory=False, **kwargs)
    valid_loader = DataLoader(dset_valid, batch_size=args.batch_size, shuffle=(args.train_type == 'random'), num_workers=args.nbworkers, pin_memory=False, **kwargs)
    test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=(args.train_type == 'random'), num_workers=args.nbworkers, pin_memory=False, **kwargs)
    return train_loader, valid_loader, test_loader, args

def get_external_sounds(path, ref_loader, args, **kwargs):
    dset = AudioDataset(path, data=args.data, mean=ref_loader.means, var=ref_loader.vars, **kwargs)
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.nbworkers, pin_memory=True, **kwargs)
    dset.final_params = ref_loader.final_params
    return loader

if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--path', type=str, default='/Users/esling/Datasets/diva_dataset', help='')
    parser.add_argument('--dataset', type=str, default='toy', help='')
    parser.add_argument('--data', type=str, default='mel', help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    args = parser.parse_args()
    train_loader, valid_loader, test_loader, args = load_dataset(args)
    # Take fixed batch (train)
    data, params, meta, audio = next(iter(train_loader))
    plot_batch(data[:16].unsqueeze(1))
    plot_batch_detailed(data[:5], params[:5])
    # Take fixed batch (train)
    data, params, meta = next(iter(test_loader))
    plot_batch(data[:16].unsqueeze(1))
    plot_batch_detailed(data[:5], params[:5])