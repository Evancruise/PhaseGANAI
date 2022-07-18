import h5py
import torch
import numpy as np
from pathlib import Path
from torch.utils import data

class Dataset2channel(data.Dataset):
    def __init__(self, file_path, phase, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.p = Path(file_path)
        self.phase = phase
        self.total_num_train = len(sorted(self.p.glob('*.h5')))
        self.total_num_test = len(sorted(self.p.glob('**/*.h5')))
        assert (self.p.is_dir())
        if recursive:
            #print('p.glob("**/*.h5"):', p.glob('**/*.h5'))
            #files = sorted(p.glob('**/*.h5'))
            print('Total number of images:', len(sorted(self.p.glob('**/*.h5'))))
            files = sorted(self.p.glob('**/*.h5'))
        else:
            #print('p.glob("*.h5"):', p.glob('*.h5'))
            #files = sorted(p.glob('*.h5'))
            print('Total number of images:', len(sorted(self.p.glob('*.h5'))))
            files = sorted(self.p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
    
    def __getitem__(self, index):
        
        if self.phase == 'train':
            total_num = self.total_num_train
        else:   
            total_num = self.total_num_test

        x = self.get_data("phi", index)
        x = torch.from_numpy(x)
        
        y = self.get_data("A", index)
        y = torch.from_numpy(y)

        z1 = self.get_data("I1", index)
        z1 = torch.from_numpy(z1)
        
        z2 = self.get_data("I2", index)
        z2 = torch.from_numpy(z2)
        
        e = self.get_data("phi_proj", index)
        e = torch.from_numpy(e)
        
        f = self.get_data("A_proj", index)
        f = torch.from_numpy(f)
        
        z1_0 = self.get_data("I1o", index)
        z1_0 = torch.from_numpy(z1_0)
        
        z2_0 = self.get_data("I2o", index)
        z2_0 = torch.from_numpy(z2_0)
        
        index_number = self.get_data("i", index)
        
        hypothesis = self.get_data("hypothesis", index)
        
        image_type = self.get_data("image_type", index)
        
        recon_alg = self.get_data("recon_alg", index)
        
        return (x, y, z1, z2, e, f, z1_0, z2_0, index_number, hypothesis, total_num, image_type, recon_alg)
        # return (x, y, z2, z1, a, b, c, d, e, f, U, index_number)

    def __len__(self):
        return len(self.get_data_infos('phi'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path,'r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    idx = -1
                    if load_data:
                        idx = self._add_to_cache(ds.value, file_path)
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'cache_idx': idx})

    def _load_data(self, file_path):
        with h5py.File(file_path,'r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    idx = self._add_to_cache(ds[()], file_path)
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)
                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]
