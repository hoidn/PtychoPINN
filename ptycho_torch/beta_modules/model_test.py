#Dump file for random modules that I've tried out

class ResBlock(nn.Module):
    """Simple Residual Block."""
    def __init__(self, channels, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels) if batch_norm else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Add skip connection *before* final activation
        out += identity
        # Apply activation after skip connection
        out = self.relu(out) # Or another activation if preferred
        return out

#This decoder just doesnt seem to work that well unfortunately. Most of the info seems to be in the core of the object, with the higher frequencies
#not that informative
class Decoder_last(nn.Module):
    '''
    Base class for the final decoder stage. Handles conditional BatchNorm
    and enhanced x2 path processing.
    '''
    def __init__(self, model_config: ModelConfig, data_config: DataConfig,
                 in_channels, out_channels,
                 activation = torch.sigmoid, name = '', batch_norm=False):
        super(Decoder_last, self).__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.n_filters_scale = model_config.n_filters_scale
        self.N = self.data_config.N
        self.gridsize = self.data_config.grid_size

        # Get parameters from config for x2 path
        self.x2_in_channels = model_config.decoder_x2_channels
        self.x2_intermediate_channels = model_config.decoder_x2_intermediate_channels

        # Ensure x2_in_channels is not more than the total input channels
        if self.x2_in_channels >= in_channels:
            print(f"Warning: decoder_x2_channels ({self.x2_in_channels}) >= in_channels ({in_channels}). Adjusting x2_in_channels.")
            # Adjust x2_in_channels, e.g., to half or a fixed smaller number
            self.x2_in_channels = max(4, in_channels // 4) # Example adjustment

        self.x1_in_channels = in_channels - self.x2_in_channels

        # --- Layers for Path 1 (x1) ---
        self.conv1 =  nn.Conv2d(in_channels = self.x1_in_channels,
                                out_channels = out_channels,
                                kernel_size = (3, 3),
                                padding = 3//2)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity() # Use Identity if BN is off
        self.activation = activation # Activation for x1 path


        # --- Layers for Path 2 (x2) ---
        # ConvUpBlock now takes x2_in_channels and outputs x2_intermediate_channels
        self.conv_up_block = ConvUpBlock(self.x2_in_channels, self.x2_intermediate_channels,
                                         batch_norm = batch_norm) # Pass flag

        # Add ResBlock for increased depth/complexity
        self.res_block = ResBlock(self.x2_intermediate_channels, batch_norm=batch_norm)

        # Final Conv for x2 path, maps from intermediate channels to output channels
        self.conv2 =  nn.Conv2d(in_channels = self.x2_intermediate_channels,
                                out_channels = out_channels,
                                kernel_size = (3, 3),
                                padding = 3//2)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity() # Use Identity if BN is off
        # Activation for x2 path remains SiLU/Swish
        self.activation_x2 = F.silu


        # --- Common Layers ---
        self.batch_norm = batch_norm # Store for reference if needed, already used in bn layers
        self.padding = nn.ConstantPad2d((self.N // 4, self.N // 4,
                                         self.N // 4, self.N //4), 0)


    def forward(self,x):
        # Split input based on the configured number of channels for x2
        x_for_x1 = x[:, :self.x1_in_channels, :, :]
        x_for_x2 = x[:, -self.x2_in_channels:, :, :]

        # --- Path 1 (x1) ---
        x1 = self.conv1(x_for_x1)
        x1 = self.bn1(x1) # Apply BN before activation
        x1 = self.activation(x1)
        x1 = self.padding(x1) # Padding applied only to x1

        # --- Path 2 (x2) ---
        # Only compute if probe_big is True (as per original logic)
        if not self.model_config.probe_big:
            # If probe isn't big, only x1 contributes (as per original padding logic)
            # Ensure the output shape matches expectation if probe_big=False
            # The original code returned only x1. Let's stick to that.
             return x1

        x2 = self.conv_up_block(x_for_x2)
        x2 = self.res_block(x2) # Apply ResBlock
        x2 = self.conv2(x2)
        x2 = self.bn2(x2) # Apply BN before final activation
        x2 = self.activation_x2(x2) # Apply SiLU

        # Add the outputs of the two paths
        outputs = x1 + x2

        return outputs