import numpy as np
from ptycho.params import params, get
from torch.utils.data import Dataset
from mmap_ninja import RaggedMmap
from pathlib import Path
import zipfile
from collections import defaultdict
import torch
import time

#Memory mapping
from tensordict import MemoryMappedTensor, TensorDict
from tensordict.prototype import tensorclass

#Patch generation
from ptycho_torch.patch_generator import group_coords, get_relative_coords

#Helper methods
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

    Params
    grid_size = tuple of image grid size (e.g. 2 x 2 is most used)
                n_images = grid_size[0] * grid_size[1] (e.g. 2 x 2 = 4)
    probe_map = list of probe indices assigned to each experiment
                (e.g. [0, 1, 0] -> Exp 1, Probe 1. Exp 2, Probe 2. Exp 3, Probe 1)
    probes = list of probes used in experiments. Will be h x w numpy tensors.
    K = # of nearest neighbors to group together to select image patch from
    n_subsample = # of patches to subsample from each group of size K
        e.g. K = 6, n_subsample = 10, n_images = 4.  Then we subsample 10 patches from 6C4=15

    """
    def __init__(self, ptycho_dir, probe_dir, params):
        #Directories
        self.ptycho_dir = ptycho_dir
        self.probe_dir = probe_dir

        #Note to self: probe mapping will be in params
        self.params = params

        #Calculate length for __len__ method
        self.length, self.im_shape, self.cum_length = self.calculate_length(ptycho_dir)

        #Putting all relevant information within accessible dictionary
        self.data_dict = defaultdict(list)
        #Image stack
        self.memory_map_data(Path(self.ptycho_dir).iterdir())

        
    def calculate_length(self, ptycho_dir):
        """
        Calculates length from series of npz files without loading them using npz_header
        Also calculates cumulative length for linear indexing accounting for n_subsample
        """
        total_length = 0
        cumulative_length = [0]

        for npz_file in Path(ptycho_dir).iterdir():
            tensor_shape = list(npz_headers(npz_file))
            total_length += tensor_shape[0][0] * self.params['n_subsample'] #Double indexing to access number inside nested list
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
        n_images = self.params['grid_size'][0] * self.params['grid_size'][1]
        self.params['n_images'] = n_images

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
                    (mmap_length, n_images, *self.im_shape),
                    dtype=torch.float64,
                ),
                "coords_center": MemoryMappedTensor.empty(
                    (mmap_length, 1, 1, 2),
                    dtype=torch.float64,
                ),
                "coords_relative": MemoryMappedTensor.empty(
                    (mmap_length, n_images, 1, 2),
                    dtype=torch.float64,
                ),
                "coords_start_center": MemoryMappedTensor.empty(
                    (mmap_length, 1, 1, 2),
                    dtype=torch.float64,
                ),
                "coords_start_relative": MemoryMappedTensor.empty(
                    (mmap_length, n_images, 1, 2),
                    dtype=torch.float64,
                ),
                "nn_indices": MemoryMappedTensor.empty(
                    (mmap_length, n_images),
                    dtype=torch.int64
                )},
            batch_size = mmap_length,
        )

        #End timer
        end = time.time()
        print("Memory map creation time: {}".format(end - start))

        #Lock memory map
        mmap_ptycho.memmap_(prefix="data/memmap")

        #Go through each npz file and populate mmap_diffraction
        batch_size = 1024
        #Keep track of memory map write indices
        global_from, global_to = 0, 0

        for i, npz_file in enumerate(image_paths):
            print("Populating memory map for dataset {}".format(i))
            #NON-DIFFRACTION IMAGE MAPPING
            #----
            #Assume: N = # of scans
            xcoords = np.load(npz_file)['xcoords']
            ycoords = np.load(npz_file)['ycoords']
            xcoords_start = np.load(npz_file)['xcoords_start']
            ycoords_start = np.load(npz_file)['ycoords_start']

            start, end = self.cum_length[i], self.cum_length[i+1]

            #Get indices for solution patches on current dataset
            nn_indices, coords_nn = group_coords(xcoords, ycoords,
                                      self.params, C=self.params['C'])
            
            #Coords_nn is (N x 4 x 1 x 2). Contains all 4 sets of (x,y) coords for an image patch
            coords_start_nn = np.stack([xcoords_start[nn_indices],
                                        ycoords_start[nn_indices]],axis=2)[:, :, None, :]
            
            #Get relative and center of mass coordinates
            coords_com, coords_relative = get_relative_coords(coords_nn)
            coords_start_com, coords_start_relative = get_relative_coords(coords_start_nn)

            #Writing to non-diffraction memory maps in one go:
            non_diff_timer_start = time.time()

            mmap_ptycho["coords_center"][start:end] = torch.from_numpy(coords_com)
            mmap_ptycho["coords_relative"][start:end] = torch.from_numpy(coords_relative)
            mmap_ptycho["coords_start_center"][start:end] = torch.from_numpy(coords_start_com)
            mmap_ptycho["coords_start_relative"][start:end] = torch.from_numpy(coords_start_relative)
            mmap_ptycho["nn_indices"][start:end] = torch.from_numpy(nn_indices)

            non_diff_time = time.time() - non_diff_timer_start
            print("Non-diffraction memory map write time: {}".format(non_diff_time))

            #DIFFRACTION IMAGE MAPPING
            #----
            diff_timer_start = time.time()
            curr_nn_index_length = len(nn_indices)
            diff_stack = np.load(npz_file)['diff3d']

            #Write to memory mapped tensor in batches
            for j in range(0, curr_nn_index_length, batch_size):
                #Calculate end index (to not exceed length of list)
                local_to = min(j + batch_size, curr_nn_index_length)
                global_to += local_to - j

                print('global_from: {}, global_to: {}'.format(global_from, global_to))
                print('j: {}, local_to: {}'.format(j, local_to))
  
                #Write to diffraction memory map
                mmap_ptycho["images"][global_from:global_to] = torch.from_numpy(diff_stack[nn_indices[j:local_to]])

                #Update global
                global_from += global_to - global_from

            diff_time = time.time() - diff_timer_start
            print("Diffraction memory map write time: {}".format(diff_time))

        self.mmap_ptycho = mmap_ptycho

        return 
    
    def generate_ptycho_groupings(self, index):
        """
        Generates ptycho grouping indices using methods from patch_generator
        """
        nn_indices, coords_nn = group_coords(self, index)

        return nn_indices, coords_nn



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
        Returns memory mapped tensordict, alongside probe
        """
        #Find experimental index comparing index to cumulative length
        exp_idx = np.searchsorted(self.cum_length, idx, side='right') - 1
        probe_idx = self.params['probe_map'][exp_idx]
        probe = torch.from_numpy(self.params['probes'][probe_idx])

        return self.mmap_ptycho[idx], probe
        



#Testing
#Going to implement tensorclass which has memory mapping TENSOR functionality built in, will be built on top of 
#existing dataset

#Tensor memory mapping has the advantage that it's already in shared memory so using multiple workers for the
#dataloader on a single gpu is viable

#Dataset will contain naive getitem methods