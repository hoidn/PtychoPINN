#Torch
import torch
from torch import nn
import torch.nn.functionl as F

#Other math
import numpy as np
import math

#Helper
from ptycho_torch.config_params import Config
from ptycho_torch.config_params import Params
import ptycho_torch.helper as hh

#Helping modules

class Tanh_custom_act(nn.Module):
    '''
    Custom tanh activation module used in:
        Decoder_phase
    '''
    def forward(self, x):
        return math.pi * torch.tanh(x)

#Conv blocks
class ConvBaseBlock(nn.Module):
    '''
    Convolutional base block for Pooling and Upscaling
    '''
    def __init__(self, n_channels,
                 w1 = 3, w2 = 3,
                 padding = 'same',
                 activation = 'relu'):
        super(ConvBaseBlock, self).__init__()
        padding_size = w1 // 2 if padding == 'same' else 0
        #NN layers
        self.conv1 = nn.Conv2d(in_filters = n_channels,
                               out_filters = n_channels,
                               kernel_size = (w1, w2),
                               padding = padding_size)
        self.conv2 = nn.Conv2d(in_filters = n_channels,
                               out_filters = n_channels,
                               kernel_size = (w1, w2),
                               padding = padding_size)
        #Activation used in upblock
        self.activation = getattr(F, activation) if activation else None
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.activation(self.conv2(x)) if self.activation else F.relu(self.conv2(x))
        x = self._pool_or_up(x)

        return x
    
    def _pool_or_up(self, x):
        raise NotImplementedError("Subclasses must implement pool or up")
    
class ConvPoolBlock(ConvBaseBlock):

    def __init__(self, n_channels,
                 w1 = 3, w2 = 3, p1 = 2, p2 = 2,
                 padding = 'same'):
        super(ConvPoolBlock, self).__init__(n_channels, w1, w2, padding)
        padding_size = w1 // 2 if padding == 'same' else 0
        #Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(p1, p2),
                                 padding = padding_size if padding == 'same' else 0)
        
    def _pool_or_up(self, x):
        return self.pool(x)
    
class ConvUpBlock(ConvBaseBlock):

    def __init__(self, n_channels,
                 w1 = 3, w2 = 3, p1 = 2, p2 = 2,
                 mode = 'nearest', padding = 'same'):
        
        super(ConvUpBlock, self).__init__(n_channels, w1, w2, padding)
        padding_size = w1 // 2 if padding == 'same' else 0
        #NN layers
        self.up = nn.Upsample(scale_factor = (p1, p2),
                              mode = 'nearest')

    def _pool_or_up(self, x):
        return self.up(x)

#Encoder

class Encoder(nn.Module):
    def __init__(self, n_filters_scale):
        super(Encoder, self).__init__()

        self.N = Config().get('N')

        starting_coeff = 64 / (self.N / 32)
        self.filters = [int(n_filters_scale * starting_coeff * 2**i) for i in range(4)]

        if starting_coeff < 3 or starting_coeff > 64:
            raise ValueError(f"Unsupported input size: {self.N}")
        
        self.blocks = nn.ModuleList([ConvPoolBlock(num_filters)] 
                                    for num_filters in self.filters)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x
    
#Decoders

class Decoder_filters(nn.Module):
    def __init__(self, n_filters_scale):
        super(Decoder_filters, self).__init__()
        self.N = Config().get('N')

        #Calculate number of channels for upscaling
        #Start from self.N and divide by 2 until 32 for each layer
        #E.g. 
        #N == 64: [n_filters_scale * 64, n_filters_scale * 32]
        #N == 128: [n_filters_scale * 128, n_filters_scale * 64, n_filters_scale * 32]
        n_terms = int(np.log(self.N / 32) / np.log(2) + 1)
        self.filters = [int(n_filters_scale * self.N * (1/2)**i) for i in range(n_terms)]

        if self.N < 32:
            raise ValueError(f"Unsupported input size: {self.N}")
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward")
    
class Decoder_base(Decoder_filters):
    def __init__(self, n_filters_scale):
        super(Decoder_base, self).__init__()
        #Layers
        self.blocks = nn.ModuleList([ConvUpBlock(num_filters)] 
                                    for num_filters in self.filters)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class Decoder_last(nn.Module):
    '''
    Decoder to get both amplitude and phase, combining to create full object function
    '''
    def __init__(self, n_filters_scale,
                  conv1, conv2,
                  activation = torch.sigmoid, name = ''):
        super(Decoder_last, self).__init__()
        #Grab parameters
        self.N = Config().get('N')
        self.gridsize = Config().get('gridsize')

        #Layers
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv_up_block = ConvUpBlock(n_filters_scale * 32)

        #Additional
        self.activation = activation
        self.padding = nn.ConstantPad2d((self.N // 4, self.N // 4,
                                         self.N // 4, self.N //4), 0)
        self.c_outer = 4
        
    def forward(self,x):
        x1 = self.conv1(x[:, :-self.c_outer, :, :])
        x1 = self.activation(x1)
        x1 = self.padding(x1)

        if not Config().get('probe.big'):
            return x1
        
        x2 = self.conv_up_block(x[:, -self.c_outer:, :, :])
        x2 = self.conv2(x2)
        x2 = F.silu(x2) #Same as swish

        outputs = x1 + x2

        return outputs

class Decoder_phase(Decoder_base):
    def __init__(self, n_filters_scale):
        super(Decoder_phase, self).__init__()
        #Nn layers
        conv1 = nn.Conv2d(in_filters = n_filters_scale,
                            out_filters = n_filters_scale,
                            kernel_size = (3, 3),
                            padding = 'same')
        conv2 = nn.Conv2d(in_filters = n_filters_scale,
                            out_filters = n_filters_scale,
                            kernel_size = (3, 3),
                            padding = 'same')
        #Custom nn layers with specific identifiable names
        self.add_module('phase_activation', Tanh_custom_act())
        self.add_module('phase', Decoder_last(n_filters_scale,
                                         conv1, conv2,
                                         activation = self.phase_activation))
            
    def forward(self, x):
        #Apply upscale block layers
        for block in self.blocks:
            x = block(x)

        #Apply final layer
        outputs = self.phase(x)

class Decoder_amp(Decoder_base):
    def __init__(self, n_filters_scale):
        super(Decoder_amp, self).__init__()
        #Nn layers
        conv1 = nn.Conv2d(in_filters = 1,
                            out_filters = 1,
                            kernel_size = (3, 3),
                            padding = 'same')
        conv2 = nn.Conv2d(in_filters = 1,
                            out_filters = 1,
                            kernel_size = (3, 3),
                            padding = 'same')
        #Custom nn layers with specific identifiable names
        self.add_module('amp_activation', Tanh_custom_act())
        self.add_module('amp', Decoder_last(n_filters_scale,
                                         conv1, conv2,
                                         activation = self.amp_activation))
            
    def forward(self, x):
        #Apply upscale block layers
        for block in self.blocks:
            x = block(x)

        #Apply final layer
        outputs = self.amp(x)

        return outputs

#Autoencoder

class Autoencoder(nn.Module):
    def __init__(self, n_filters_scale):
        super(Autoencoder, self).__init__()
        #Encoder
        encoder = Encoder(n_filters_scale)
        #Decoders (Amplitude/Phase)
        decoder_amp = Decoder_amp(n_filters_scale)
        decoder_phase = Decoder_phase(n_filters_scale)

    def forward(self, x):
        #Encoder
        x = self.encoder(x)
        #Decoders
        x_amp = self.decoder_amp(x)
        x_phase = self.decoder_phase(x)

        return x_amp, x_phase


#Other modules
class CombineComplex(nn.Module):
    '''
    Converts real number amplitude and phase into single complex number for FFT

    Inputs
    ------
    amp: torch.Tensor
        Amplitude of complex number
    phi: torch.Tensor
        Phase of complex number
    
    Outputs
    -------
    out: torch.Tensor
        Complex number
    '''
    def __init__(self):
        super(CombineComplex, self).__init__()

    def forward(self, amp: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:

        out = amp.to(dtype=torch.complex64) * \
                torch.exp(1j * phi.to(dtype=torch.complex64))
        
        return out
    
#Scaling modules

# # Example usage
# # Assuming cfg is a dictionary with the key 'intensity_scale' and appropriate value
# cfg = {'intensity_scale': 1.0}
# params = lambda: {'intensity_scale.trainable': True}

# # Initialize the module
# scaler_module = IntensityScalerModule(cfg, params)

# # Create instances of the classes
# scaler = scaler_module.IntensityScaler(scaler_module.log_scale)
# inv_scaler = scaler_module.IntensityScalerInv(scaler_module.log_scale)

# # Example input tensor
# input_tensor = torch.randn(4, 4)

# # Apply scaling and inverse scaling
# scaled_tensor = scaler(input_tensor)
# inv_scaled_tensor = inv_scaler(input_tensor)

class IntensityScalerModule:
    def __init__(self, cfg, params):
        #Setting log scale values
        log_scale_guess = np.log(Config().get('intensity_scale'))
        self.log_scale = nn.Parameter(torch.tensor(float(log_scale_guess)),
                                      requires_grad = Params.get('intensity_scale_trainable'))
    
    #Intensity scaler as class
    class IntensityScaler(nn.Module):
        '''
        Scales intensity with log scale factor. Supports inverse and regular scaling
        '''
        def __init__(self, log_scale, inv = False):
            super(IntensityScalerModule.IntensityScaler, self).__init__()
            self.scale_factor = torch.exp(log_scale)
            if inv:
                self.scale_factor = 1 / self.scale_factor

        def forward(self, x):
            return x / self.scale_factor
        
    #Intensity scalar as function

    def scale(self, x):
        return x / torch.exp(self.log_scale)
    
    def inv_scale(self, x):
        return x * torch.exp(self.log_scale)

#Full module with everything
class PtychoPINN(nn.Module):
    def __init__(self, cfg, params):
        #Configuration 
        self.n_filters_scale = Config().get('n_filters_scale')
        self.N = Config().get('N')
        self.gridsize = Config().get('gridsize')
        self.offset = Config().get('offset')
        self.object_big = Config().get('object.big')

        #To-do, probe

        self.autoencoder = Autoencoder(self.n_filters_scale)
        self.combine_complex = CombineComplex()

    def forward(self, x, positions, probe):
        #Autoencoder result
        x_amp, x_phase = self.autoencoder(x)
        #Combine amp and phase
        x_combined = self.combine_complex(x_amp, x_phase)

        #Reassemble patches
        if self.object_big:
            reassembled_obj = hh.reassemble_patches_position_real(
                x_combined,
                positions
            )
        else:
            #NOTE for albert: Check transformation math
            reassembled_obj = hh.pad_patches(
                torch.flatten(x_combined, start_dim = 0, end_dim = 1),
                padded_size = hh.get_padded_size()
            )

        
