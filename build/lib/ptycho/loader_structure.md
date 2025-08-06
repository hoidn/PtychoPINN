graph TD
    RawData[RawData]
    PtychoDataset[PtychoDataset]
    PtychoDataContainer[PtychoDataContainer]
    load[load]
    
    RawData --> |uses| generate_grouped_data
    RawData --> |uses| from_coords_without_pc
    RawData --> |uses| from_simulation
    RawData --> |uses| to_file
    RawData --> |uses| from_file
    RawData --> |uses| from_files
    RawData --> |uses| _check_data_validity
    
    generate_grouped_data --> |calls| get_neighbor_diffraction_and_positions
    
    PtychoDataContainer --> |contains| X
    PtychoDataContainer --> |contains| Y_I
    PtychoDataContainer --> |contains| Y_phi
    PtychoDataContainer --> |contains| norm_Y_I
    PtychoDataContainer --> |contains| YY_full
    PtychoDataContainer --> |contains| coords_nominal
    PtychoDataContainer --> |contains| coords_true
    
    load --> |uses| normalize_data
    load --> |uses| crop
    load --> |uses| get_gt_patch
    load --> |uses| get_image_patches
    
    subgraph Helper Functions
        get_neighbor_self_indices
        get_neighbor_indices
        sample_rows
        get_relative_coords
        crop12
        extract_and_translate_patch_np
        unsqueeze_coords
        calculate_combined_offsets
    end
    
    shift_and_sum
    reassemble_position --> |uses| shift_and_sum
    
    group_coords --> |uses| get_neighbor_self_indices
    group_coords --> |uses| get_neighbor_indices
    
    calculate_relative_coords --> |uses| group_coords
