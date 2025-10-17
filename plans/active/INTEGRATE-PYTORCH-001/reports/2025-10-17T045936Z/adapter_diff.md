diff --git a/ptycho/config/config.py b/ptycho/config/config.py
index 4a7d352..4661c53 100644
--- a/ptycho/config/config.py
+++ b/ptycho/config/config.py
@@ -266,24 +266,29 @@ def dataclass_to_legacy_dict(obj: Any) -> Dict[str, Any]:
 
 def update_legacy_dict(cfg: Dict[str, Any], dataclass_obj: Any) -> None:
     """Update legacy dictionary with dataclass values.
-    
+
     ⚠️ CRITICAL: Call this BEFORE any data loading operations!
-    
+
     Common failure scenario:
-    - Symptom: Shape (*, 64, 64, 1) instead of (*, 64, 64, 4) with gridsize=2  
+    - Symptom: Shape (*, 64, 64, 1) instead of (*, 64, 64, 4) with gridsize=2
     - Cause: This function wasn't called before generate_grouped_data()
     - Fix: Call immediately after config setup, before load_data()
-    
+
     Updates values from the dataclass, but skips None values to preserve
     existing parameter values when new configuration doesn't specify them.
-    
+
     Args:
         cfg: Legacy dictionary to update
         dataclass_obj: Dataclass instance containing new values
     """
     new_values = dataclass_to_legacy_dict(dataclass_obj)
-    
+
     # Update values from dataclass, but skip None values to preserve existing params
+    # Convert any remaining Path objects to strings for legacy compatibility
     for key, value in new_values.items():
         if value is not None:
-            cfg[key] = value
+            # Convert Path to string if not already done by KEY_MAPPINGS
+            if isinstance(value, Path):
+                cfg[key] = str(value)
+            else:
+                cfg[key] = value
diff --git a/ptycho_torch/config_bridge.py b/ptycho_torch/config_bridge.py
index a754443..f0061f8 100644
--- a/ptycho_torch/config_bridge.py
+++ b/ptycho_torch/config_bridge.py
@@ -141,6 +141,14 @@ def to_model_config(
         )
     amp_activation = activation_mapping[model.amp_activation]
 
+    # Translate probe_mask from Optional[Tensor] to bool
+    # None → False (no masking), non-None tensor → True (masking enabled)
+    # Can be overridden explicitly via overrides dict
+    probe_mask_value = False  # Default when None
+    if TORCH_AVAILABLE and model.probe_mask is not None:
+        # If torch available and probe_mask is a tensor, enable masking
+        probe_mask_value = True
+
     # Build kwargs from PyTorch configs
     # CRITICAL: Only include fields that exist in TensorFlow ModelConfig
     # intensity_scale_trainable belongs in TrainingConfig, NOT ModelConfig
@@ -156,14 +164,16 @@ def to_model_config(
         'object_big': model.object_big,
         'probe_big': model.probe_big,
 
+        # Translated values
+        'probe_mask': probe_mask_value,  # PyTorch Optional[Tensor] → TensorFlow bool
+
         # Default values for fields missing in PyTorch (spec-required)
-        'probe_mask': False,  # PyTorch has Optional[Tensor], TensorFlow has bool
         'pad_object': True,   # Missing in PyTorch, use TensorFlow default
         'probe_scale': data.probe_scale,  # PyTorch default=1.0, TensorFlow default=4.0
         'gaussian_smoothing_sigma': 0.0,  # Missing in PyTorch
     }
 
-    # Apply overrides
+    # Apply overrides (allows explicit probe_mask override)
     kwargs.update(overrides)
 
     return TFModelConfig(**kwargs)
@@ -215,7 +225,7 @@ def to_training_config(
 
         # From DataConfig
         'neighbor_count': data.K,    # Semantic mapping
-        'nphotons': data.nphotons,
+        'nphotons': data.nphotons,  # Will be validated after overrides
 
         # From PyTorch ModelConfig (belongs in TrainingConfig in TensorFlow)
         'intensity_scale_trainable': pt_model.intensity_scale_trainable,
@@ -240,10 +250,24 @@ def to_training_config(
     # Apply overrides (critical for MVP fields)
     kwargs.update(overrides)
 
-    # Convert string paths to Path objects if needed
+    # Validate nphotons: PyTorch default (1e5) differs from TensorFlow default (1e9)
+    # Require explicit override to avoid silent divergence (spec §5.2:9 HIGH risk)
+    pytorch_default_nphotons = 1e5
+    tensorflow_default_nphotons = 1e9
+    if 'nphotons' not in overrides and data.nphotons == pytorch_default_nphotons:
+        raise ValueError(
+            f"nphotons default divergence detected: PyTorch default ({pytorch_default_nphotons}) "
+            f"differs from TensorFlow default ({tensorflow_default_nphotons}). "
+            f"Provide explicit nphotons override to resolve: "
+            f"overrides=dict(..., nphotons={tensorflow_default_nphotons})"
+        )
+
+    # Convert string paths to Path objects, then back to strings for params.cfg compatibility
+    # This ensures KEY_MAPPINGS in config/config.py correctly converts to strings
     for path_field in ['train_data_file', 'test_data_file', 'output_dir']:
-        if path_field in kwargs and isinstance(kwargs[path_field], str):
-            kwargs[path_field] = Path(kwargs[path_field])
+        if path_field in kwargs and kwargs[path_field] is not None:
+            if isinstance(kwargs[path_field], str):
+                kwargs[path_field] = Path(kwargs[path_field])
 
     # Validate required overrides with actionable error messages
     if kwargs['train_data_file'] is None:
@@ -304,10 +328,12 @@ def to_inference_config(
     # Apply overrides (critical for MVP fields)
     kwargs.update(overrides)
 
-    # Convert string paths to Path objects if needed
+    # Convert string paths to Path objects, then back to strings for params.cfg compatibility
+    # This ensures KEY_MAPPINGS in config/config.py correctly converts to strings
     for path_field in ['model_path', 'test_data_file', 'output_dir']:
-        if path_field in kwargs and isinstance(kwargs[path_field], str):
-            kwargs[path_field] = Path(kwargs[path_field])
+        if path_field in kwargs and kwargs[path_field] is not None:
+            if isinstance(kwargs[path_field], str):
+                kwargs[path_field] = Path(kwargs[path_field])
 
     # Validate required overrides with actionable error messages
     if kwargs['model_path'] is None:
