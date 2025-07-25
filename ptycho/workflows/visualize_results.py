"""
Workflow for visualizing ptychographic reconstruction results as heatmap images.

Transforms reconstruction results and test data into heatmaps via evaluation.summarize,
saving PNG files for visual assessment of reconstruction quality.

**Input:** results dict with 'pred_amp'/'reconstructed_obj', PtychoDataContainer test_data
**Output:** PNG heatmap files at {output_prefix}/{heatmap_name}.png
**Key params:** i (sample index), output_prefix (save directory)

```python
visualize_results(results, test_data, i=200, output_prefix='analysis')
```
"""

import numpy as np
import matplotlib.pyplot as plt
from ptycho import evaluation, params
from typing import Dict, Any

def visualize_results(results: Dict[str, Any], test_data, i: int = 200, output_prefix: str = 'output'):
    """
    Visualize the results using the evaluation.summarize function.

    Args:
    results (Dict[str, Any]): Dictionary containing the results from the CDI process.
    test_data: The test data used for evaluation.
    i (int): Index of the sample to visualize. Default is 200.
    output_prefix (str): Directory to save the output files. Default is 'output'.
    """
    # Extract necessary data from results and test_data
    pred_amp = results['pred_amp']
    reconstructed_obj = results['reconstructed_obj']
    X_test = test_data.X
    Y_I_test = test_data.Y_I
    Y_phi_test = test_data.Y_phi
    probe = np.absolute(params.get('probe')[:, :, 0, 0])

    # Call the summarize function
    heatmaps = evaluation.summarize(i, results['pred_amp'] + 1, results['reconstructed_obj'], 
                                    X_test, Y_I_test, Y_phi_test,
                                    probe, channel=0, crop=False)

    # Save the heatmaps
    for name, heatmap in heatmaps.items():
        plt.figure(figsize=(10, 10))
        plt.imshow(heatmap, cmap='jet')
        plt.colorbar()
        plt.title(name)
        plt.savefig(f"{output_prefix}/{name}.png")
        plt.close()

    print(f"Heatmaps saved to {output_prefix}")

if __name__ == "__main__":
    # This is where you would load your results and test_data
    # For example:
    # from ptycho.workflows.components import load_and_prepare_data
    # test_data = load_and_prepare_data("path_to_test_data.npz")
    # results = ... # Load your results here

    # visualize_results(results, test_data)
    pass  # Remove this line when uncommenting the code above
