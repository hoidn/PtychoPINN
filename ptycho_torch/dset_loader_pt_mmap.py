import numpy as np
from ptycho.params import params, get
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import zipfile
from collections import defaultdict
import torch
import time
import os

#Memory mapping
from tensordict import MemoryMappedTensor, TensorDict

#Patch generation
from ptycho_torch.patch_generator import group_coords, get_relative_coords

#Parameters
from ptycho_torch.config_params import TrainingConfig, DataConfig, ModelConfig

#Helper methods
import ptycho_torch.helper as hh

def npz_headers(npz):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape).
    We can use this to determin shape of npz constituents without loading them
    This will be useful in the __len__ method for the dataset

    Taken from: https://stackoverflow.com/questions/68224572/how-to-determine-the-shape-size-of-npz-file
    Modified to quickly grab dimension we care about
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            #If name starts with 'xcoords'
            if name.startswith('diff3d'):

                npy = archive.open(name)
                version = np.lib.format.read_magic(npy)
                shape, _, _ = np.lib.format._read_array_header(npy, version)
                yield shape
                break



class PtychoDataset(Dataset):
    """
    Ptychography Dataset for PtychoPINN

    Important: Some data is memory-mapped in order to provide fast loading for dynamic data
    #Memory-mapped data: Diffraction images
    #Non-memory-mapped data: Probe, ObjectGuess, scan_index, coords (x,y)

    The layout of the data will be such that the first index is always the experiment #.
    For example, diffraction[0] will return the entire image stack from the first experiment
    coords[0] would return the stack of image coordinates from the first experiment

    Inputs
    -------
    ptycho_dir: Directory containing individual ptychography scans as npz files. If non-npz, expected to be normalized or
    rewritten via a data adapting software such as Ptychodus
    probe_dir: Directory containing probe guesses as npz files. If non-npz, expected to be normalized or
    rewritten via a data adapting software such as Ptychodus

    Params
    grid_size: tuple of image grid size (e.g. 2 x 2 is most used)
                n_images = grid_size[0] * grid_size[1] (e.g. 2 x 2 = 4)
    probe_map: list of probe indices assigned to each experiment
                (e.g. [0, 1, 0] -> Exp 1, Probe 1. Exp 2, Probe 2. Exp 3, Probe 1)
    probes: list of probes used in experiments. Will be h x w numpy tensors.
    K: # of nearest neighbors to group together to select image patch from
    n_subsample: # of patches to subsample from each group of size K
        e.g. K = 6, n_subsample = 10, n_images = 4.  Then we subsample 10 patches from 6C4=15

    """
    def __init__(self, ptycho_dir, probe_dir, data_dir='data/memmap', remake_map = True):
        #Directories
        self.ptycho_dir = ptycho_dir
        self.n_files = len(os.listdir(ptycho_dir))

        #Putting all relevant information within accessible dictionary
        self.data_dict = {}

        #Either grab probes from directory or expect them to be provided in configs
        if DataConfig().get('probe_dir_get'):
            self.get_probes(probe_dir)
        else:
            self.data_dict['probes'] = DataConfig().get('probes')

        #Calculate length for __len__ method and __get__
        self.length, self.im_shape, self.cum_length = self.calculate_length(ptycho_dir)

        #Image stack memory mapping
        data_prefix = data_dir.split('/')[0]
        state_path = data_prefix +'/' + 'state_files.npz'

        #If memory map wasn't made or you want to remake it
        if not os.path.exists(data_dir) or remake_map:
            self.memory_map_data(Path(self.ptycho_dir).iterdir())
            #Save self.data_dict and self.probes in npz file
            np.savez(state_path,
                     data_dict=self.data_dict)
        #Otherwise, if path exists, load memory map and probe/other constants
        else:
            #THIS FUNCTIONALITY CURRENTLY DOESNT WORK
            #I DONT EVEN KNOW IF YOU CAN LOAD A PREVIOUS MEMORY MAP
            print('Existing map found. Loading memory-mapped data')
            self.mmap_ptycho = TensorDict.load_memmap(data_dir)
            temp = np.load(state_path, allow_pickle=True)

            #Assign other important params
            self.data_dict = temp['data_dict'].item()
        
    def calculate_length(self, ptycho_dir):
        """
        Calculates length from series of npz files without loading them using npz_header
        Also calculates cumulative length for linear indexing, since we are creating "groups" of channels for ptychography training.
        This means that the total length of image groups is equal to the # diffraction images * # of groupings.

        """
        total_length = 0
        cumulative_length = [0]

        for npz_file in Path(ptycho_dir).iterdir():
            #Check whether object.big is checked
            if ModelConfig().get('object.big'):
                multiplier = DataConfig().get('n_subsample')
            else:
                multiplier = 1
            tensor_shape = list(npz_headers(npz_file))

            total_length += tensor_shape[0][0] * multiplier #Double indexing to access number inside nested list
            im_shape = tensor_shape[0][1:]
            cumulative_length.append(total_length)
        
        return total_length, im_shape, cumulative_length
    
    #Methods for diffraction data mapping
    def memory_map_data(self, image_paths):
        """
        Creates memory mapped tensor dictionary containg diffraction images and relevant coordinate information
        1.  Solves for solution patch indices using group_coords method
        2.  Writes to respective memory maps. The diffraction map is populated in batches, while the other maps
        are populated in full for every individual dataset
            - "images" - (N x C x H x W), N = # of patterns, C = # of images per soln patch, H = height, W = width
            - "coords_offsets" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "coords_relative" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "coords_start_offsets" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "coords_start_relative" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
        Note: Probe is stored in memory for the dataset, not in the memory map.
        ---
        Args:
            image_paths - list of paths to independent experiment npz files
            grid_size - tuple of image grid size (e.g. 2 x 2 is most used)


        """
        if ModelConfig().get('object.big'):
            n_channels = DataConfig().get('grid_size')[0] * DataConfig().get('grid_size')[1]
        else:
            n_channels = 1
        DataConfig().add('n_images', n_channels)

        #Create memory map for every tensor. We'll be populating the diffraction image in batches, and the
        #other coordinate tensors in full for every individual dataset

        mmap_length = self.length

        #Time creation of tensordict with printed messages
        print("Creating memory mapped tensor dictionary...")
        print("Memory map length: {}".format(mmap_length))

        #Start timer
        start = time.time()

        mmap_ptycho = TensorDict(
            {   "images": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, *self.im_shape),
                    dtype=torch.float32,
                ),
                "coords_center": MemoryMappedTensor.empty(
                    (mmap_length, 1, 1, 2),
                    dtype=torch.float32,
                ),
                "coords_relative": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, 1, 2),
                    dtype=torch.float32,
                ),
                "coords_start_center": MemoryMappedTensor.empty(
                    (mmap_length, 1, 1, 2),
                    dtype=torch.float32,
                ),
                "coords_start_relative": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, 1, 2),
                    dtype=torch.float32,
                ),
                "nn_indices": MemoryMappedTensor.empty(
                    (mmap_length, n_channels),
                    dtype=torch.int64
                ),
                "experiment_id": MemoryMappedTensor.empty(
                    (mmap_length),
                    dtype=torch.int32
                )},
            batch_size = mmap_length,
        )

        #End timer
        end = time.time()
        print("Memory map creation time: {}".format(end - start))

        #Lock memory map
        mmap_ptycho.memmap_like(prefix="data/memmap")

        #Create normalization tensor for all experiments
        self.data_dict["scaling_constant"] = torch.empty(self.n_files,
                                                         dtype = torch.float32)

        #Go through each npz file and populate mmap_diffraction
        batch_size = 1024
        #Keep track of memory map write indices
        global_from, global_to = 0, 0

        for i, npz_file in enumerate(image_paths):
            print("Populating memory map for dataset {}".format(i))
            #NON-DIFFRACTION IMAGE MAPPING
            #----
            #Assume: N = # of scans
            start, end = self.cum_length[i], self.cum_length[i+1]

            #Writing to non-diffraction memory maps in one go:
            non_diff_timer_start = time.time()

            #If solution patches are enforced, then must include alot more metadata
            if ModelConfig().get('object.big'):
                #Grabbing information from npz file
                xcoords = np.load(npz_file)['xcoords']
                ycoords = np.load(npz_file)['ycoords']
                xcoords_start = np.load(npz_file)['xcoords_start']
                ycoords_start = np.load(npz_file)['ycoords_start']
                #Get indices for solution patches on current dataset
                nn_indices, coords_nn = group_coords(xcoords, ycoords,
                                        C=DataConfig().get('C'))
                
                #Coords_nn is (N x 4 x 1 x 2). Contains all 4 sets of (x,y) coords for an image patch
                coords_start_nn = np.stack([xcoords_start[nn_indices],
                                            ycoords_start[nn_indices]],axis=2)[:, :, None, :]
                
                #Get relative and center of mass coordinates
                coords_com, coords_relative = get_relative_coords(coords_nn)
                coords_start_com, coords_start_relative = get_relative_coords(coords_start_nn)

                mmap_ptycho["coords_center"][start:end] = torch.from_numpy(coords_com)
                mmap_ptycho["coords_relative"][start:end] = torch.from_numpy(coords_relative)
                mmap_ptycho["coords_start_center"][start:end] = torch.from_numpy(coords_start_com)
                mmap_ptycho["coords_start_relative"][start:end] = torch.from_numpy(coords_start_relative)
                mmap_ptycho["nn_indices"][start:end] = torch.from_numpy(nn_indices)
            else:
                #Otherwise, the indices are just an arange from 0 to N-1
                nn_indices = np.arange(end-start)
                mmap_ptycho["nn_indices"][start:end] = torch.from_numpy(nn_indices)[:,None]

            mmap_ptycho["experiment_id"][start:end] = torch.tensor(i)

            non_diff_time = time.time() - non_diff_timer_start
            print("Non-diffraction memory map write time: {}".format(non_diff_time))

            #DIFFRACTION IMAGE MAPPING
            #----
            diff_timer_start = time.time()
            curr_nn_index_length = len(nn_indices)
            diff_stack = torch.from_numpy(np.load(npz_file)['diff3d'])

            #Inserting dummy channel dimension if n_channels = 1
            if not ModelConfig().get('object.big'):
                diff_stack = diff_stack[:,None]

            #Perform normalization on diffraction image stack
            if DataConfig().get('normalize'):
                diff_stack, norm_factor = hh.normalize_data(diff_stack)
                self.data_dict["scaling_constant"][i] = norm_factor
            else:
                self.data_dict["scaling_constant"][i] = 1

            #Write to memory mapped tensor in batches
            for j in range(0, curr_nn_index_length, batch_size):
                #Calculate end index (to not exceed length of list)
                local_to = min(j + batch_size, curr_nn_index_length)
                global_to += local_to - j

                mmap_ptycho["images"][global_from:global_to] = diff_stack[nn_indices[j:local_to]]

                #Writing everything simultaneously as a TensorDict
                # mmap_ptycho[global_from:global_to] = TensorDict(
                #     {
                #         "images": torch.from_numpy(diff_stack[nn_indices[j:local_to]]),
                #         "coords_center": torch.from_numpy(coords_com[j:local_to]),
                #         "coords_relative": torch.from_numpy(coords_relative[j:local_to]),
                #         "coords_start_center": torch.from_numpy(coords_start_com[j:local_to]),
                #         "coords_start_relative": torch.from_numpy(coords_start_relative[j:local_to]),
                #         "nn_indices": torch.from_numpy(nn_indices[j:local_to]),
                #         "experiment_id": torch.from_numpy(np.full(local_to - j, i))
                #     },
                #     batch_size = (local_to - j)
                # )

                #Update global
                global_from += global_to - global_from

            diff_time = time.time() - diff_timer_start
            print("Diffraction memory map write time: {}".format(diff_time))

            #Calculate intensity scale for given experiment and add to 

        

        self.mmap_ptycho = mmap_ptycho

        return 
    
    def get_probes(self, probe_dir):
        '''
        Retrieve all used experimental probes from probe directory. Expect a series of npz files with different probes.
        Puts them all in the same tensor, self.probes, which will be indexed in __getitem__

        '''
        N = DataConfig().get('N')
        n_probes = len(os.listdir(probe_dir))
        self.data_dict['probes'] = torch.empty(size=(n_probes, N, N), dtype = torch.complex64)
        print(os.listdir(probe_dir))
        for i, probe_file in enumerate(os.listdir(probe_dir)):
            probe_path = os.path.join(probe_dir, probe_file)
            probe_data = np.load(probe_path)
            probe_data = torch.from_numpy(probe_data['probe'])
            self.data_dict['probes'][i] = probe_data

    def load_diffraction_data(self, npz_file):
        """
        Loads diffraction image stack from npz file. Used for memory mapping only
        (don't use for get_item)
        ---
        Input: npz_file - path to npz file
        Output: diffraction - (N x H x W), N = # of patterns, H = height, W = width
        """
        diffraction = np.load(npz_file)['diff3d']
        return diffraction

    #Methods for mapping all other data 
    def map_data(self, ptycho_dir):
        """
        Currently unused.

        Maps all data from npzs to specific arrays held in memory
        Data will be in format data_dict['parameter'][experiment #][image #]
        """

        #Take above code and instead use a for loop and unpack for each variable
        for npz_file in Path(ptycho_dir).iterdir():
            temp_file = np.load(npz_file)
            #Append all relevant information
            self.data_dict['xcoords'].append(temp_file['xcoords'])
            self.data_dict['ycoords'].append(temp_file['ycoords'])
            self.data_dict['xstart'].append(temp_file['xcoords_start'])
            self.data_dict['ystart'].append(temp_file['ycoords_start'])
            
            self.data_dict['probeGuess_global'].append(temp_file['probeGuess']) #Note: this is a 2D array

        return 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns memory mapped tensordict, alongside probe. Written to be batched
        so you can return multiple instances.

        Probe dimensionality is expanded to match the data channels. This is so multiplication operations are broadcast correctly.
        
        Output
        -------
        self.mmap_ptych[idx] - Batched TensorDict containing all relevant information for training. See
            function memory_map_data for further details. Length is batch size
        probes_indexed - (N,C,H,W) tensor, where N is batch size. C is number of channels, where the probe is duplicated
            H and W are height and width of diffraction pattern. The dimensionality should be exactly the same as the output of the autoencoder.
        scaling_constant - (N) tensor, scaling constants required for each diffraction image
        
        """
        #Experimental index is used to find the probe corresponding to the right experiment
        #exp_idx is a list of experiment indices that are then used to index probe/scaling constants
        exp_idx = self.mmap_ptycho['experiment_id'][idx]

        if ModelConfig().get('object.big'):
            channels = DataConfig().get('C')
        else:
            channels = 1

        #Expand probe to match number of channels for data.
        probes_indexed = self.data_dict['probes'][exp_idx].unsqueeze(1).expand(-1,channels,-1,-1)

        #Scaling constant
        scaling_const = self.data_dict['scaling_constant'][exp_idx]

        return self.mmap_ptycho[idx], probes_indexed, scaling_const
        
#Collation

class TensorDictDataLoader(DataLoader):
    '''
    Modifiers dataloader class that allows for batch sampling exploiting the structure of TensorDicts
    Given a set of indices, we can directly index all of them simultaneously from the TensorDict instead of calling
    yield on a single index at a time.

    This allows us to return a TensorDict object which already has indexing built in.
    '''
    def __iter__(self):
        #Iterator over sampler
        batch_sampler = self.batch_sampler
        dataset = self.dataset

        for batch_indices in batch_sampler:
            yield dataset[batch_indices]



#Tensor memory mapping has the advantage that it's already in shared memory so using multiple workers for the
#dataloader on a single gpu is viable

#Dataset will contain naive getitem methods