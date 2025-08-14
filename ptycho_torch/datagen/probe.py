
# Artifical probes here

import numpy as np
import scipy.special
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod

# Math modules here:
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq # Keep fft imports in case needed later
import scipy.ndimage as ndi
import scipy.linalg # For orthogonalization
import random
from scipy.special import erf
from scipy.ndimage import gaussian_filter

# Define a type alias for clarity, assuming it means np arrays of floats
RealArrayType = np.ndarray

# --- Zernike Polynomial Definition (Adapted from provided class) ---
@dataclass(frozen=True)
class ZernikePolynomial:
    """
    Represents a single Zernike polynomial defined by radial degree (n)
    and angular frequency (m), normalized such that integral(Z^2) = pi.
    The polynomial is defined within the unit disk (distance <= 1).
    Uses standard (Noll/Born&Wolf) indexing convention implicitly via n, m.
    """
    radial_degree: int  # n >= 0
    angular_frequency: int  # m, |m| <= n, n-|m| is even

    # Basic validation
    def __post_init__(self):
        if self.radial_degree < 0:
            raise ValueError("Radial degree (n) cannot be negative.")
        if abs(self.angular_frequency) > self.radial_degree:
            raise ValueError("Absolute angular frequency (|m|) cannot exceed radial degree (n).")
        if (self.radial_degree - abs(self.angular_frequency)) % 2 != 0:
            raise ValueError("The difference n - |m| must be even.")

    @property
    def noll_index(self) -> Optional[int]:
        """Calculate the Noll index j (starting from j=1). Optional, not strictly needed for generation."""
        # This is a common but sometimes confusing index. Calculation can be complex.
        # For generation based on (n,m), we don't strictly need it.
        # Implementation Reference: e.g., https://wp.optics.arizona.edu/visualopticslab/wp-content/uploads/sites/52/2021/08/Zernike-Polynomials-Wyant.pdf
        # Or https://en.wikipedia.org/wiki/Noll_index
        # Let's skip implementing this for now unless specifically required.
        return None

    def _radial_polynomial(self, distance: RealArrayType) -> RealArrayType:
        """Calculates the radial part R_n^|m|(rho)."""
        n = self.radial_degree
        m_abs = abs(self.angular_frequency)
        n_minus_m = n - m_abs
        half_n_minus_m = n_minus_m // 2

        # Ensure distance is clipped to [0, 1] for calculation stability if needed,
        # although the __call__ method handles the domain.
        rho = np.clip(distance, 0.0, 1.0)
        radial_poly = np.zeros_like(rho)

        # Summation formula for R_n^|m|
        for k in range(half_n_minus_m + 1):
            sign = (-1)**k
            numerator = np.math.factorial(n - k)
            denominator = (np.math.factorial(k) *
                           np.math.factorial(half_n_minus_m - k) *
                           np.math.factorial((n + m_abs) // 2 - k)) # (n+m)/2 - k

            # Handle potential precision issues with large factorials if n is high
            # scipy.special.binom might be more stable
            # Be cautious if n > ~20

            term_coeff = sign * numerator / denominator
            radial_poly += term_coeff * np.power(rho, n - 2 * k)

        return radial_poly


    def _angular_function(self, angle: RealArrayType) -> RealArrayType:
        """Calculates the angular part cos(m*theta) or sin(|m|*theta)."""
        m = self.angular_frequency
        if m == 0:
            # Factor of sqrt(1) if m=0
            return np.ones_like(angle) # Constant angular dependence
        elif m > 0:
            # Factor of sqrt(2) included in normalization below
            return np.cos(m * angle)
        else: # m < 0
            # Factor of sqrt(2) included in normalization below
            return np.sin(abs(m) * angle)

    def __call__(
        self, distance: RealArrayType, angle: RealArrayType, undefined_value: float = 0.0
    ) -> RealArrayType:
        """
        Evaluates the Zernike polynomial Z_n^m(rho, theta) on the unit disk.

        Args:
            distance (RealArrayType): Radial coordinate (rho), normalized to [0, 1].
            angle (RealArrayType): Angular coordinate (theta) in radians.
            undefined_value (float): Value outside the unit disk (distance > 1).

        Returns:
            RealArrayType: The value of the Zernike polynomial at the given coordinates.
                           Includes standard normalization factor sqrt(2(n+1)) or sqrt(n+1).
        """
        # Check if n and m are valid (done in __post_init__)
        n = self.radial_degree
        m = self.angular_frequency

        # Calculate radial and angular parts
        radial_part = self._radial_polynomial(distance)
        angular_part = self._angular_function(angle)

        # Apply normalization factor sqrt(2*(n+1)) for m!=0, sqrt(n+1) for m=0
        norm_factor = np.sqrt(n + 1.0)
        if m != 0:
            norm_factor *= np.sqrt(2.0)

        # Combine parts and apply normalization
        zernike_value = norm_factor * radial_part * angular_part

        # Apply the unit disk mask
        mask = (distance >= 0) & (distance <= 1)
        return np.where(mask, zernike_value, undefined_value)

    def __str__(self) -> str:
        return f'Z(n={self.radial_degree}, m={self.angular_frequency})'


# --- Function to Generate a Single Zernike Probe ---

def generate_zernike_probe(
    shape: Tuple[int, int],
    diameter_pixels: float,
    probe_arg,
    max_order: int = 10,
    coeff_mean: complex = 0.0 + 0.0j,
    coeff_stdev: float = 0.1,
    sparsity: float = 0.1,
    fix_piston: Optional[complex] = 1.0 + 0.0j,
    seed: Optional[int] = None,
    ) -> np.ndarray:
    """
    Generates a single complex probe based on a random combination of Zernike
    polynomials, defining the probe aperture diameter directly in pixels.

    Args:
        shape (Tuple[int, int]): Output probe shape (height, width).
        diameter_pixels (float): Diameter of the probe's circular support
                                 in pixel units. Zernike polynomials are
                                 defined within this disk.
        max_order (int): Maximum radial degree 'n' for Zernike polynomials
                         to include in the basis. Must be >= 0.
        coeff_mean (complex): Mean value for the random complex coefficients.
        coeff_stdev (float): Standard deviation for the real and imaginary parts
                             of the random complex coefficients (drawn from Gaussian).
        sparsity (float): Probability (0.0 to 1.0) that a given Zernike coefficient
                          (excluding piston if fixed) will be forced to zero.
        fix_piston (Optional[complex]): If not None, the coefficient for Z(n=0, m=0)
                                        (piston term) is fixed to this value instead
                                        of being randomized. Defaults to 1+0j.
        normalize (bool): If True, normalize the final probe so that sum(|probe|^2) = 1.
        seed (Optional[int]): Seed for the random number generator for reproducibility.

    Returns:
        np.ndarray: A complex-valued 2D np array representing the generated probe.
    """
    if max_order < 0:
        raise ValueError("max_order must be non-negative.")
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError("sparsity must be between 0.0 and 1.0")
    if diameter_pixels <= 0:
        raise ValueError("diameter_pixels must be positive.")

    height, width = shape
    radius_pixels = diameter_pixels / 2.0

    # Optional warning if diameter is larger than grid
    if diameter_pixels > min(width, height):
         print(f"Warning: diameter_pixels ({diameter_pixels}) is larger than the "
               f"smallest grid dimension ({min(width, height)}). The probe will fill the disk region within the grid.")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # --- 1. Generate Coordinate System in PIXEL Units ---
    center_y_pix, center_x_pix = (height - 1) / 2.0, (width - 1) / 2.0
    # Create pixel coordinate grids relative to the center
    y_pix = (np.arange(height) - center_y_pix)
    x_pix = (np.arange(width) - center_x_pix)
    xx_pix, yy_pix = np.meshgrid(x_pix, y_pix, indexing='xy') # Match Zernike angle convention

    # Calculate distance from center in PIXEL units
    distance_pixels = np.sqrt(xx_pix**2 + yy_pix**2)

    # Calculate angle (same as before, depends on relative coordinates)
    angle_rad = np.arctan2(yy_pix, xx_pix)

    # --- 2. Normalize distance by PIXEL radius ---
    # This gives rho, the normalized radial coordinate for Zernike evaluation
    # Avoid division by zero if radius_pixels is extremely small, though check prevents zero.
    distance_norm = distance_pixels / radius_pixels

    # --- 3. Generate Zernike Basis up to max_order ---

    max_order = max_order
    # (This part is identical to the previous function)
    zernike_basis: List[ZernikePolynomial] = []
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            try:
                poly = ZernikePolynomial(radial_degree=n, angular_frequency=m)
                zernike_basis.append(poly)
            except ValueError as e:
                 print(f"Skipping invalid Zernike index (n={n}, m={m}): {e}")
                 continue
    if not zernike_basis:
        print("Warning: No valid Zernike polynomials generated.")
        return np.zeros(shape, dtype=complex)

    # --- 4. Generate Random Coefficients ---
    # (This part is identical to the previous function)
    num_coeffs = len(zernike_basis)
    coefficients = np.zeros(num_coeffs, dtype=complex)
    for i, poly in enumerate(zernike_basis):
        if poly.radial_degree == 0 and poly.angular_frequency == 0 and fix_piston is not None:
            coefficients[i] = fix_piston; continue
        real_part = np.random.normal(loc=coeff_mean.real, scale=coeff_stdev)
        imag_part = np.random.normal(loc=coeff_mean.imag, scale=coeff_stdev)
        random_coeff = real_part + 1j * imag_part
        if np.random.rand() < sparsity: coefficients[i] = 0.0 + 0.0j
        else: coefficients[i] = random_coeff

    # --- 5. Combine Zernike Polynomials ---
    # (This part is identical, uses distance_norm and angle_rad)
    probe_array = np.zeros(shape, dtype=complex)
    for i, poly in enumerate(zernike_basis):
        if coefficients[i] == 0j: continue
        # Evaluate polynomial on the grid using normalized distance & angle
        zernike_values = poly(distance_norm, angle_rad, undefined_value=0.0)
        probe_array += coefficients[i] * zernike_values # zernike_values is real

    return probe_array


def generate_random_zernike(shape, probe_arg):
    """
    Wrapper function to randomize inputs going into zernike generator
    """ 
    IMG_SHAPE = shape
    DIAMETER_PIXELS = np.random.randint(10,25)
    MAX_Z_ORDER = 10        # Max n for Zernike basis (e.g., up to Z_5^m)
    COEFF_STDEV = 0.5     # Spread of random coefficients
    SPARSITY = 0.1         # ~40% of coefficients (excluding piston) set to zero
    SIGMA_X = (DIAMETER_PIXELS//2) * np.random.uniform(0.3,0.5)
    SIGMA_Y = SIGMA_X# + np.random.uniform(0.95,1.05)

    zernike_probe = generate_zernike_probe(
        shape=IMG_SHAPE,
        diameter_pixels= DIAMETER_PIXELS,
        probe_arg = probe_arg,
        max_order=MAX_Z_ORDER,
        coeff_stdev=COEFF_STDEV,
        sparsity=SPARSITY,
        fix_piston=1.0 + 0.0j, # Ensure main mode has reasonable amplitude
        seed=np.random.randint(0,1e3) # Use different seed for each initial probe
    )

    # Step 2: Apply Gaussian envelope

    zernike_probe = apply_gaussian_envelope(zernike_probe,
                                          SIGMA_X,
                                          SIGMA_Y,
                                          normalize_output = True)
    
    return zernike_probe

# --- Example Usage ---
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     # --- Parameters ---
#     IMG_SHAPE = (128, 128)
#     PIXEL_SIZE = 5e-9      # 5 nm
#     PROBE_DIAMETER = 100e-9 # 100 nm diameter aperture for Zernikes
#     MAX_Z_ORDER = 5        # Max n for Zernike basis (e.g., up to Z_5^m)
#     COEFF_STDEV = 0.15     # Spread of random coefficients
#     SPARSITY = 0.4         # ~40% of coefficients (excluding piston) set to zero
#     NUM_INITIAL_PROBES = 5 # Generate 5 random probes initially
#     NUM_MODES_FINAL = 3    # Keep the top 3 orthogonal modes after SVD

#     # --- Generate Multiple Initial Probes ---
#     print(f"Generating {NUM_INITIAL_PROBES} initial random Zernike probes...")
#     initial_probes = []
#     for i in range(NUM_INITIAL_PROBES):
#         probe = generate_zernike_probe(
#             shape=IMG_SHAPE,
#             pixel_size=PIXEL_SIZE,
#             diameter_m=PROBE_DIAMETER,
#             max_order=MAX_Z_ORDER,
#             coeff_stdev=COEFF_STDEV,
#             sparsity=SPARSITY,
#             fix_piston=1.0 + 0.0j, # Ensure main mode has reasonable amplitude
#             normalize=True,        # Normalize each initial probe
#             seed=i # Use different seed for each initial probe
#         )
#         initial_probes.append(probe)
#     print("Done generating initial probes.")



#--- FZP probe ---#

def generate_orthogonal_modes_from_primary(
    primary_probe: np.ndarray,
    num_modes_total: int,
    seed: Optional[int] = None
    ) -> List[np.ndarray]:
    """
    Generates a set of orthogonal modes derived from a single primary probe.

    The primary probe is included as the basis for the first mode. Subsequent
    initial modes are generated by applying random linear phase ramps to the
    primary probe. The full set is then orthogonalized using scipy.linalg.orth.

    Args:
        primary_probe (np.ndarray): The complex 2D input probe array.
        num_modes_total (int): The total number of orthogonal modes desired
                               (including the one based on the primary probe).
                               Must be >= 1.
        seed (Optional[int]): Seed for the random number generator used for
                              phase ramps.

    Returns:
        List[np.ndarray]:
            A list containing the orthogonalized complex probe modes (2D arrays).
            The number of modes returned is min(num_modes_total, rank), where rank
            is the numerical rank determined by the orthogonalization process.
            The first mode in the list is the component corresponding most closely
            to the normalized primary probe. All modes are normalized (sum(|mode|^2)=1).
            Returns an empty list if primary_probe is invalid or num_modes_total < 1.
    """
    if not np.iscomplexobj(primary_probe) or primary_probe.ndim != 2:
        raise ValueError("primary_probe must be a 2D complex np array.")
    if num_modes_total < 1:
        print("Warning: num_modes_total must be at least 1. Returning empty list.")
        return []

    # Normalize the input primary probe first (defensively)
    norm_primary = np.linalg.norm(primary_probe.ravel())
    if norm_primary < 1e-12:
        print("Warning: Primary probe has near-zero norm. Cannot generate modes.")
        return []
    primary_probe_normalized = primary_probe / norm_primary

    if num_modes_total == 1:
        return [primary_probe_normalized]

    shape = primary_probe.shape
    height, width = shape
    vector_len = height * width

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # --- 1. Initialize list with the primary mode ---
    initial_modes_flat = [primary_probe_normalized.ravel()]

    # --- 2. Generate derived modes with random phase ramps ---
    # Create coordinate ramps for phase generation, from -1 to 1
    y_ramp = np.linspace(-1.0, 1.0, height)
    x_ramp = np.linspace(-1.0, 1.0, width)
    xx_ramp, yy_ramp = np.meshgrid(x_ramp, y_ramp, indexing='xy')

    for i in range(num_modes_total - 1):
        # Generate random coefficients for the linear ramp ax + by
        # Adjust range if ramps need to be stronger/weaker
        a = random.uniform(-1.0, 1.0)
        b = random.uniform(-1.0, 1.0)

        # Create phase ramp: exp(i * pi * (a*x + b*y))
        phase_ramp = np.exp(1j * np.pi * (a * xx_ramp + b * yy_ramp))

        # Create derived mode
        derived_mode = primary_probe_normalized * phase_ramp
        initial_modes_flat.append(derived_mode.ravel())

    # --- 3. Orthogonalize the set ---
    # Create matrix A with flattened modes as columns
    A = np.stack(initial_modes_flat, axis=-1) # Shape (vector_len, num_modes_total)

    # Use scipy.linalg.orth to find an orthonormal basis for the column space
    # This performs the orthogonalization using SVD internally by default.
    # It returns a matrix whose columns are the orthonormal basis vectors.
    try:
        orthonormal_matrix = scipy.linalg.orth(A) # Shape (vector_len, rank)
        rank = orthonormal_matrix.shape[1]
        if rank < num_modes_total:
            print(f"Warning: Initial modes were linearly dependent. Found {rank} orthogonal modes instead of {num_modes_total}.")

    except Exception as e:
         print(f"Error during orthogonalization: {e}")
         return []

    # --- 4. Reshape and return ---
    orthogonal_modes = []
    for i in range(rank):
        mode_flat = orthonormal_matrix[:, i]
        mode_2d = mode_flat.reshape(shape)
        orthogonal_modes.append(mode_2d)

    return orthogonal_modes

## Usage
# --- Parameters ---
# IMG_SHAPE = (128, 128)
# FZP_DIAM_PIX = 64.0       # FZP diameter in pixels
# ZONE_WIDTH_PIX = 1.5     # Outermost zone width in pixels
# BEAMSTOP_PIX = 10.0      # Beamstop diameter in pixels
# DEFOCUS_STRENGTH = 1.2   # Apply some defocus phase

# NUM_ORTHO_MODES = 4      # Request 4 orthogonal modes

# # --- Step 1: Generate the primary FZP probe ---
# print("Generating primary FZP probe...")
# primary_fzp = generate_fzp_probe_pixel_units(
#     shape=IMG_SHAPE,
#     fzp_diameter_pix=FZP_DIAM_PIX,
#     outermost_zone_width_pix=ZONE_WIDTH_PIX,
#     beamstop_diameter_pix=BEAMSTOP_PIX,
#     defocus_phase_strength=DEFOCUS_STRENGTH
# )
# print("Primary probe generated.")

# # --- Step 2: Generate orthogonal modes from the primary probe ---
# print(f"\nGenerating {NUM_ORTHO_MODES} orthogonal modes from primary...")
# orthogonal_modes = generate_orthogonal_modes_from_primary(
#     primary_probe=primary_fzp,
#     num_modes_total=NUM_ORTHO_MODES,
#     seed=42
# )
# print(f"Generated {len(orthogonal_modes)} orthogonal modes.")



#--- Experimental ---

@dataclass
class PropagatorParameters:
    wavelength_m: float
    width_px: int
    height_px: int
    pixel_width_m: float
    pixel_height_m: float
    propagation_distance_m: float

    @property
    def dx(self) -> float:
        """pixel width in wavelengths"""
        return self.pixel_width_m / self.wavelength_m

    @property
    def pixel_aspect_ratio(self) -> float:
        """pixel aspect ratio (width / height)"""
        return self.pixel_width_m / self.pixel_height_m

    @property
    def z(self) -> float:
        """propagation distance in wavelengths"""
        return self.propagation_distance_m / self.wavelength_m

    @property
    def fresnel_number(self) -> float:
        """fresnel number"""
        return np.square(self.dx) / np.absolute(self.z)

    def get_spatial_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        JJ, II = np.mgrid[: self.height_px, : self.width_px]
        XX = II - self.width_px // 2
        YY = JJ - self.height_px // 2
        return YY.astype(float), XX.astype(float)

    def get_frequency_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        fx = fftshift(fftfreq(self.width_px))
        fy = fftshift(fftfreq(self.height_px))
        FY, FX = np.meshgrid(fy, fx, indexing='ij')
        return FY, FX

# --- Provided/Required Helper Classes: Propagator & FresnelTransformPropagator ---
class Propagator(ABC):
    @abstractmethod
    def propagate(self, wavefield):
        pass

class FresnelTransformPropagator(Propagator):
    def __init__(self, parameters: PropagatorParameters) -> None:
        self._params = parameters # Store for reference if needed

        ipi = 1j * np.pi
        Fr = parameters.fresnel_number
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px
        M = parameters.height_px
        YY, XX = parameters.get_spatial_coordinates()

        # Term C0: Proportional to Fr / (i * ar)
        C0 = (Fr / (1j * ar))

        C1 = np.exp(2j * np.pi * parameters.z)  # Phase factor exp(ikz)
        C2 = np.exp((np.square(XX / N) + np.square(ar * YY / M)) * ipi / Fr)

        is_forward = parameters.propagation_distance_m >= 0.0
        
        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0
        self._B = np.exp(ipi * Fr * (np.square(XX) + np.square(YY / ar)))


    def propagate(self, wavefield):
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield * self._B)))
        else:
            return self._B * fftshift(ifft2(ifftshift(wavefield * self._A)))


# --- Minimal Internal FresnelZonePlate class for physical parameters ---
class _FresnelZonePlateInternal:
    def __init__(self, zone_plate_diameter_m: float, outermost_zone_width_m: float, central_beamstop_diameter_m: float):
        if zone_plate_diameter_m < 0:
            raise ValueError("Zone plate diameter cannot be negative.")
        if zone_plate_diameter_m > 0 and outermost_zone_width_m <= 0:
            raise ValueError("Outermost zone width must be positive for a physical FZP with non-zero diameter.")
        if central_beamstop_diameter_m < 0:
            raise ValueError("Central beamstop diameter cannot be negative.")
        
        self.zone_plate_diameter_m = zone_plate_diameter_m
        self.outermost_zone_width_m = outermost_zone_width_m
        self.central_beamstop_diameter_m = central_beamstop_diameter_m

    def get_focal_length_m(self, wavelength_m: float) -> float:
        if wavelength_m <= 0:
            raise ValueError("Wavelength must be positive.")
        return (self.zone_plate_diameter_m * self.outermost_zone_width_m) / wavelength_m

# --- Core Function 1: Create FZP Wavefield at Optics Plane ---
def create_fzp_wavefield_at_optics_plane(
    # FZP physical parameters
    zone_plate_diameter_m: float,
    outermost_zone_width_m: float,
    central_beamstop_diameter_m: float,
    transition_width: float,
    # Illumination and geometry parameters
    wavelength_m: float, #Pretty low, just check this
    array_width_px: int, #This is just the probe width we care about (i.e. 64 pixels)
    array_height_px: int, #This is just the probe width we care about (i.e. 64 pixels)
    buffer: int,
    # Defocus: distance from FZP focal plane to the *intended sample plane*
    defocus_distance_m: float, #800 micron, ranges from 300-1000 micron
    # Desired pixel size at the final sample plane (used to calculate optics plane pixel size)
    sample_plane_pixel_width_m: float, #Typically 5-10 nm, this is the max resolution
    sample_plane_pixel_height_m: float, #Typically 5-10 nm, this is the max resolution
    # Aberrations (optional)
    phase_noise_std_rad: Optional[float] = None,
    amplitude_noise_std: Optional[float] = None, # Standard deviation relative to max ideal FZP amplitude
    zernike_dict: Optional[Dict] = None, # List of (ZernikePolynomial, amplitude_in_radians)
):
    """
    Creates a Fresnel Zone Plate (FZP) wavefield at the optics (FZP) plane.

    Returns:
        Tuple: (wavefield_at_optics_plane, optics_plane_pixel_width_m, 
                optics_plane_pixel_height_m, propagation_distance_to_sample_m)
    """
    if wavelength_m <= 0: raise ValueError("Wavelength must be positive.")
    if array_width_px <= 0 or array_height_px <= 0: raise ValueError("Array dimensions must be positive.")
    if sample_plane_pixel_width_m <= 0 or sample_plane_pixel_height_m <= 0: raise ValueError("Sample plane pixel dimensions must be positive.")

    fzp = _FresnelZonePlateInternal(
        zone_plate_diameter_m, outermost_zone_width_m, central_beamstop_diameter_m
    )

    focal_length_m = fzp.get_focal_length_m(wavelength_m)

    # propagation_distance_to_sample_m can be positive (sample after focus/FZP) or negative (sample before focus/FZP)
    propagation_distance_to_sample_m = focal_length_m + defocus_distance_m
    
    # fzp_half_width = (array_width_px + 1) // 2
    # fzp_half_height = (array_height_px + 1) // 2

    fzp_plane_pixel_size_numerator = wavelength_m * propagation_distance_to_sample_m
    optics_plane_pixel_width_m=fzp_plane_pixel_size_numerator / (array_width_px * sample_plane_pixel_width_m)
    optics_plane_pixel_height_m=fzp_plane_pixel_size_numerator / (array_height_px * sample_plane_pixel_height_m)


    # 2. Determine total simulation array size at optics plane
    sim_array_width_px_total = array_width_px + 2 * buffer
    sim_array_height_px_total = array_height_px + 2 * buffer

    # 3. Create coordinates for this larger simulation grid
    x_coords = np.arange(sim_array_width_px_total) - np.float64(sim_array_width_px_total//2)
    y_coords = np.arange(sim_array_height_px_total) - np.float64(sim_array_height_px_total//2)

    x_coords *= optics_plane_pixel_width_m
    y_coords *= optics_plane_pixel_height_m
    # coordinate on FZP plane
    # x_coords = -optics_plane_pixel_width_m * np.arange(-fzp_half_width, fzp_half_width)
    # y_coords = -optics_plane_pixel_height_m * np.arange(-fzp_half_height, fzp_half_height)
    
    XX_FZP, YY_FZP = np.meshgrid(x_coords, y_coords, indexing = 'xy') # XX varies along cols, YY along rows
    RR_FZP = np.hypot(XX_FZP, YY_FZP)

    # FZP transmission function: T_lens * Aperture * Beamstop
    # Lens term (quadratic phase)
    phase_term = (-2 * np.pi / wavelength_m) * (XX_FZP**2 + YY_FZP**2) / (2 * focal_length_m)
    T_lens = np.exp(1j * phase_term)

    #Define aperture, beamstop edges
    aper_transition_width_m = transition_width * fzp.zone_plate_diameter_m
    aperture_radius = fzp.zone_plate_diameter_m / 2.0
    Aperture = smooth_edge(RR_FZP, aperture_radius, aper_transition_width_m)

    beamstop_transition_width_m = transition_width * fzp.central_beamstop_diameter_m
    if fzp.central_beamstop_diameter_m > 0:
        beamstop_radius = fzp.central_beamstop_diameter_m / 2.0
        # For beamstop, we want 1 outside and 0 inside, so we invert the smooth edge
        Beamstop = 1 - smooth_edge(RR_FZP, beamstop_radius, beamstop_transition_width_m)
    else:
        Beamstop = np.ones_like(RR_FZP)

    # If central_beamstop_diameter_m is 0, Beamstop is True everywhere.
    # If central_beamstop_diameter_m >= zone_plate_diameter_m, product is 0 (fully blocked or edge case).

    fzp_transmission = (T_lens * Aperture * Beamstop)#.astype(np.complex128, copy=False)
    wavefield_optics = fzp_transmission.copy() # Start with ideal FZP, then add aberrations

    #Mask for noise
    fzp_physical_mask = (RR_FZP <= aperture_radius)

    # Add Phase Noise
    if phase_noise_std_rad is not None and phase_noise_std_rad > 0:
        phase_error = np.random.normal(0, phase_noise_std_rad, size=(sim_array_height_px_total, sim_array_width_px_total))
        wavefield_optics[fzp_physical_mask] *= np.exp(1j * phase_error[fzp_physical_mask])

    # Add Amplitude Noise
    if amplitude_noise_std is not None and amplitude_noise_std > 0:
        current_amplitude = np.abs(wavefield_optics[fzp_physical_mask])
        current_phase = np.angle(wavefield_optics[fzp_physical_mask])
        
        max_ideal_amplitude = np.max(np.abs(fzp_transmission)) if np.any(fzp_transmission) else 0.0
        
        # Noise std is relative to max ideal amplitude
        actual_noise_dev = amplitude_noise_std * max_ideal_amplitude if max_ideal_amplitude > 0 else amplitude_noise_std

        if actual_noise_dev > 0: # Only add if noise deviation is meaningful
            amplitude_additive_noise = np.random.normal(0, actual_noise_dev, size=(sim_array_height_px_total, sim_array_width_px_total))
            noisy_amplitude = current_amplitude + amplitude_additive_noise[fzp_physical_mask]
            noisy_amplitude = np.maximum(noisy_amplitude, 0) # Amplitude cannot be negative
            wavefield_optics[fzp_physical_mask] = noisy_amplitude * np.exp(1j * current_phase)

    # Add Zernike Polynomial Aberrations
    if zernike_dict and fzp.zone_plate_diameter_m > 0:
        # Zernikes defined on unit disk; FZP diameter defines this disk.
        norm_radius = fzp.zone_plate_diameter_m / 2.0
        rho_coords = RR_FZP / norm_radius
        theta_coords = np.arctan2(YY_FZP, XX_FZP) # atan2(y,x)

        # Temporary views for coordinates only within the FZP mask
        rho_coords_on_fzp = rho_coords[fzp_physical_mask]
        theta_coords_on_fzp = theta_coords[fzp_physical_mask]

        total_zernike_phase = np.zeros_like(wavefield_optics[fzp_physical_mask], dtype=float)
        for (n, m), coeff in zernike_dict.items():
            zernike_poly = ZernikePolynomial(n, m)
            zernike_phase_map = zernike_poly(rho_coords_on_fzp, theta_coords_on_fzp)
            total_zernike_phase += coeff.real * zernike_phase_map
        
        wavefield_optics[fzp_physical_mask] *= np.exp(1j * total_zernike_phase)

    return (
        wavefield_optics,
        optics_plane_pixel_width_m,
        optics_plane_pixel_height_m,
        propagation_distance_to_sample_m,
    )

# --- Core Function 2: Propagate Wavefield to Object Plane ---
def propagate_to_object_plane(
    wavefield_at_optics_plane,
    wavelength_m: float,
    optics_plane_pixel_width_m: float, # Pixel size at the input wavefield plane
    optics_plane_pixel_height_m: float,
    propagation_distance_m: float, # Signed distance to propagate
    normalize_output: bool = True, # Option to normalize final wavefield
):
    """
    Propagates a wavefield from an initial plane to an object plane.

    Returns:
        Tuple: (wavefield_at_object_plane, object_plane_pixel_width_m, object_plane_pixel_height_m)
    """
    height_px, width_px = wavefield_at_optics_plane.shape

    prop_params = PropagatorParameters(
        wavelength_m=wavelength_m,
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=optics_plane_pixel_width_m,
        pixel_height_m=optics_plane_pixel_height_m,
        propagation_distance_m=propagation_distance_m,
    )
    propagator = FresnelTransformPropagator(prop_params)
    wavefield_object = propagator.propagate(wavefield_at_optics_plane)

    # Calculate effective pixel size at the object plane using Fresnel scaling:
    # dx_obj = (lambda * |z|) / (N_x * dx_optics)
    obj_pixel_width_m = (wavelength_m * abs(propagation_distance_m)) / \
                        (width_px * optics_plane_pixel_width_m)
    obj_pixel_height_m = (wavelength_m * abs(propagation_distance_m)) / \
                            (height_px * optics_plane_pixel_height_m)

    if normalize_output:
        # Normalize to sum of squared magnitudes = 1 (common for probe intensity normalization)
        total_intensity = np.sum(np.abs(wavefield_object)**2)
        if total_intensity > 1e-12: # Avoid division by zero for dark fields
            wavefield_object /= np.sqrt(total_intensity)
        # Else: wavefield is essentially zero, no normalization needed or possible.
            
    return wavefield_object, obj_pixel_width_m, obj_pixel_height_m

#FZP helper

def apply_gaussian_envelope(
    wavefield: np.ndarray,
    sigma_x_px: float = None,  # Width of Gaussian in pixels (x-direction)
    sigma_y_px: float = None,  # Width of Gaussian in pixels (y-direction)
    center_x_px: float = None, # Center x position in pixels
    center_y_px: float = None, # Center y position in pixels
    normalize_output: bool = True
) -> np.ndarray:
    """
    Apply a Gaussian envelope to a wavefield, modulating its amplitude.
    
    Args:
        wavefield: Complex wavefield to modulate
        sigma_x_px: Standard deviation in x direction (pixels)
        sigma_y_px: Standard deviation in y direction (pixels)
        center_x_px: Center x position (default: center of image)
        center_y_px: Center y position (default: center of image)
        normalize_output: Whether to normalize the final wavefield
        
    Returns:
        Wavefield with Gaussian envelope applied
    """
    height_px, width_px = wavefield.shape
    
    # Default to centered Gaussian
    if center_x_px is None:
        center_x_px = width_px // 2
    if center_y_px is None:
        center_y_px = height_px // 2
    
    # Default to isotropic Gaussian with width 1/4 of the smaller dimension
    if sigma_x_px is None:
        sigma_x_px = min(width_px, height_px) / 4
    if sigma_y_px is None:
        sigma_y_px = min(width_px, height_px) / 4
    
    # Create coordinate grids
    x = np.arange(width_px)
    y = np.arange(height_px)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Gaussian envelope
    gaussian = np.exp(
        -((X - center_x_px)**2 / (2 * sigma_x_px**2) + 
          (Y - center_y_px)**2 / (2 * sigma_y_px**2))
    )
    
    # Apply envelope to wavefield (modulates amplitude but preserves phase)
    modulated_wavefield = wavefield * gaussian
    
    # Normalize if requested
    if normalize_output:
        total_intensity = np.sum(np.abs(modulated_wavefield)**2)
        if total_intensity > 1e-12:  # Avoid division by zero
            modulated_wavefield /= np.sqrt(total_intensity)
    
    return modulated_wavefield

def smooth_edge(r, r_edge, transition_width):
    """
    Creates a smooth transition at edge r_edge with specified transition width.
    
    Args:
        r: Radial coordinate array
        r_edge: Radius at which the edge is centered
        transition_width: Width of the transition region (smaller = sharper)
        
    Returns:
        Smoothed mask with values between 0 and 1
    """
    # Use error function for smooth transition
    # erf goes from -1 to 1, so we rescale to go from 0 to 1
    return 0.5 * (1 - np.tanh((r - r_edge) / transition_width))

#FZP master function

def generate_random_fzp(shape,
                           probe_arg):
    
    #FZP parameters
    ZONE_PLATE_DIAMETER_M = np.random.randint(150, 180) * 1e-6 #(115, 180)
    OUTERMOST_ZONE_WIDTH_M = np.random.randint(30, 50) * 1e-9 #(15, 70)
    CENTRAL_BEAMSTOP_DIAMETER_M = random.uniform(0.1,0.2)* ZONE_PLATE_DIAMETER_M #(15, 80)
    TRANSITION_WIDTH = random.uniform(0.01,0.02)

    #Pixel consistency parameters
    WAVELENGTH = random.uniform(0.13,0.15) * 1e-9 #
    ARRAY_WIDTH_PX = 128
    ARRAY_HEIGHT_PX = 128
    DEFOCUS_DISTANCE_M = random.uniform(50,400) * 1e-6 #Steve uses 800
    DETECTOR_DISTANCE_M = 1.9#random.uniform(2,2.8)
    DETECTOR_PIXEL_SIZE_M = 75 * 1e-6
    SAMPLE_PLANE_PIXEL_WIDTH_M = (WAVELENGTH * DETECTOR_DISTANCE_M)/(ARRAY_WIDTH_PX * DETECTOR_PIXEL_SIZE_M)
    SAMPLE_PLANE_PIXEL_HEIGHT_M = (WAVELENGTH * DETECTOR_DISTANCE_M)/(ARRAY_HEIGHT_PX * DETECTOR_PIXEL_SIZE_M)
    BUFFER = 0
    #Aberrations
    
    PHASE_NOISE = random.uniform(0, 0.2) if probe_arg['phase_noise'] else 0  # Relative amplitude noise
    AMP_NOISE = random.uniform(0, 0.02) if probe_arg['amp_noise'] else 0  # Radians std dev phase noise
    
    #ENVELOPE
    ENVELOPE_SIGMA_W = np.random.randint(10,20)
    ENVELOPE_SIGMA_H = ENVELOPE_SIGMA_W + np.random.randint(-3,3)

    zernike_dict = {
        # (2, 0): random.uniform(-0.1,0.1) + 0j,
        # (2, 2): random.uniform(-0.05,0.05) + 0j,    # Astigmatism at 0 deg
        # (2,-2): 0.15 + 0j,  # Astigmatism at 45 deg
        # (3, -1): random.uniform(-0.02,0.02) + 0j, # Coma (y-axis)
        # (3, 3): random.uniform(-0.2,0.2) + 0j, # Coma (y-axis)
        # (3, -3): random.uniform(-0.2,0.2) + 0j, # Coma (y-axis)
        # (4, 0): random.uniform(-0.1,0.1) + 0j, #Spherical
        # (4, 2): random.uniform(-0.2,0.2) + 0j, #Secondary astigmatism
        # (4, -2): random.uniform(-0.2,0.2) + 0j
        # (5, 4): 0.1 + 0j
    }
    print(zernike_dict.items())

    #Generate wavefield

    wavefield, obj_pix_w, obj_pix_h, prop_dist = create_fzp_wavefield_at_optics_plane(
    # FZP physical parameters
    zone_plate_diameter_m = ZONE_PLATE_DIAMETER_M,
    outermost_zone_width_m = OUTERMOST_ZONE_WIDTH_M,
    central_beamstop_diameter_m = CENTRAL_BEAMSTOP_DIAMETER_M,
    transition_width=TRANSITION_WIDTH,
    # Illumination and geometry parameters
    wavelength_m = WAVELENGTH, #Pretty low, just check this
    array_width_px = ARRAY_WIDTH_PX, #This is just the probe width we care about (i.e. 64 pixels)
    array_height_px =  ARRAY_HEIGHT_PX, #This is just the probe width we care about (i.e. 64 pixels)
    buffer = BUFFER,
    # Defocus: distance from FZP focal plane to the *intended sample plane*
    defocus_distance_m = DEFOCUS_DISTANCE_M, #800 micron, ranges from 300-1000 micron
    # Desired pixel size at the final sample plane (used to calculate optics plane pixel size)
    sample_plane_pixel_width_m = SAMPLE_PLANE_PIXEL_WIDTH_M, 
    sample_plane_pixel_height_m = SAMPLE_PLANE_PIXEL_HEIGHT_M, 
    # Aberrations (optional)
    phase_noise_std_rad = PHASE_NOISE,
    amplitude_noise_std = AMP_NOISE, # Standard deviation relative to max ideal FZP amplitude
    zernike_dict = zernike_dict, # List of (ZernikePolynomial, amplitude_in_radians)
    )

    print("Propagation distance: ", prop_dist)
    print("Sample Plane Width m", SAMPLE_PLANE_PIXEL_WIDTH_M)
    print("Optics plane pixel width", obj_pix_w)
    print("Zone Plate Width", ZONE_PLATE_DIAMETER_M)
    print("Outer Zone Width", OUTERMOST_ZONE_WIDTH_M)
    print("Central beamstop diameter", CENTRAL_BEAMSTOP_DIAMETER_M)
    print("Detector distance", DETECTOR_DISTANCE_M)
    print("Wavelength", WAVELENGTH)


    #Propagate wavefield
    propagated_wavefield, _, _ = propagate_to_object_plane(wavefield,
                                                           WAVELENGTH,
                                                           obj_pix_w, obj_pix_h,
                                                           prop_dist,
                                                           normalize_output=True)
    
    # zoom_factor_h = ARRAY_HEIGHT_PX / propagated_wavefield.shape[0]
    # zoom_factor_w = ARRAY_WIDTH_PX / propagated_wavefield.shape[1]

    # propagated_wavefield = scipy.ndimage.zoom(propagated_wavefield, (zoom_factor_h, zoom_factor_w), order=2)

    #Add some additional noise

    #Final phase noise 
    final_phase_noise_mask =  np.random.normal(0, 0.15, size=(ARRAY_HEIGHT_PX, ARRAY_WIDTH_PX))
    smoothed_phase_mask = gaussian_filter(final_phase_noise_mask,1.5)
    tapered_phase_mask = apply_gaussian_envelope(smoothed_phase_mask,
                                                 sigma_x_px = 7, sigma_y_px = 7,
                                                 normalize_output = False)
    #Apply phase noise
    propagated_wavefield *= np.exp(1j * tapered_phase_mask)

    #Apply amplitude noise
    final_amp_noise_mask =  np.random.normal(0, 0.05, size=(ARRAY_HEIGHT_PX, ARRAY_WIDTH_PX))
    smoothed_amp_mask = gaussian_filter(final_amp_noise_mask,1)
    tapered_amp_mask = apply_gaussian_envelope(smoothed_amp_mask,
                                                 sigma_x_px = 3, sigma_y_px = 3,
                                                 normalize_output = False)
    #Apply amp noise
    current_amp = np.abs(propagated_wavefield)
    current_phase = np.angle(propagated_wavefield)
    propagated_wavefield = (current_amp + tapered_amp_mask) * np.exp(1j * current_phase)


    propagated_wavefield = apply_gaussian_envelope(
        propagated_wavefield,
        sigma_x_px=ENVELOPE_SIGMA_W,  # Adjust as needed for your application
        sigma_y_px=ENVELOPE_SIGMA_H,  # Adjust as needed for your application
        normalize_output=True
    )


    
    # center_height_pixel = ARRAY_HEIGHT_PX //2
    # center_width_pixel = ARRAY_WIDTH_PX // 2

    # cropped_wavefield = envelope_wavefield[center_height_pixel - shape[0]//2:center_height_pixel + shape[0]//2,
    #                                       center_width_pixel - shape[1]//2:center_width_pixel + shape[1]//2]

    return scipy.ndimage.zoom(propagated_wavefield, (shape[0] / ARRAY_HEIGHT_PX,
                                                     shape[1] / ARRAY_WIDTH_PX))#propagated_wavefield


def bin_image(image, bin_factor):
    """
    Bins an image by the specified factor in both dimensions.
    
    Parameters:
    image (numpy.ndarray): Input image as 2D array
    bin_factor (int): Binning factor (e.g., 8 to bin 512×512 to 64×64)
    
    Returns:
    numpy.ndarray: Binned image
    """
    # Get original image dimensions
    height, width = image.shape
    
    # Calculate new dimensions
    new_height = height // bin_factor
    new_width = width // bin_factor
    
    # Reshape to prepare for binning
    reshaped = image[:new_height*bin_factor, :new_width*bin_factor]  # Crop if necessary
    reshaped = reshaped.reshape(new_height, bin_factor, new_width, bin_factor)
    
    # Average pixels in each bin
    binned_image = reshaped.mean(axis=(1, 3))
    
    return binned_image
