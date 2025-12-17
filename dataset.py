"""
Dataset Class for Hypergraph Construction
PyTorch Dataset with caching and preprocessing - ROBUST VERSION
"""

from typing import List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler

from .preprocessing import AdvancedPreprocessor
from .hypergraph import HypergraphConstructor


class LungNoduleHypergraphDataset(Dataset):
    """
    PyTorch Dataset for lung nodule hypergraphs with robust file mapping.
    
    Args:
        patient_files: List of paths to patient .nii.gz files
        annotations_df: DataFrame with nodule annotations
        preprocessor: Preprocessing pipeline
        hypergraph_constructor: Hypergraph construction module
        augment: Whether to apply data augmentation
    """
    
    def __init__(
        self,
        patient_files: List[Path],
        annotations_df: pd.DataFrame,
        preprocessor: AdvancedPreprocessor,
        hypergraph_constructor: HypergraphConstructor,
        augment: bool = False
    ) -> None:
        self.annotations_df = annotations_df
        self.preprocessor = preprocessor
        self.hypergraph_constructor = hypergraph_constructor
        self.augment = augment
        self.data_cache = {}
        
        # Build stem -> path mapping for robust lookup
        self._stem_to_path = {p.stem: p for p in patient_files}
        
        # Normalize CSV patient IDs
        self.annotations_df['patient-id'] = self.annotations_df['patient-id'].astype(str).str.strip()
        
        # Build list of valid patient files that have annotations
        self.patient_files = self._build_valid_patient_list(patient_files)
        
        print(f"✓ Dataset initialized with {len(self.patient_files)} valid patients")
    
    def _resolve_patient_file(self, patient_id: str) -> Optional[Path]:
        """
        Resolve patient_id to file path using multiple strategies.
        """
        pid = str(patient_id).strip()
        
        if pid in self._stem_to_path:
            return self._stem_to_path[pid]
        
        parts = pid.split('_')
        if len(parts) > 1:
            numeric_part = parts[-1]
            try:
                num = int(numeric_part)
                for candidate in [f"{num}", f"{num:03d}", f"{num:04d}", f"{num:05d}"]:
                    for stem, path in self._stem_to_path.items():
                        if stem.endswith(candidate) or candidate in stem:
                            return path
            except ValueError:
                pass
        
        for stem, path in self._stem_to_path.items():
            if pid in stem or stem in pid:
                return path
        
        try:
            import difflib
            stems = list(self._stem_to_path.keys())
            matches = difflib.get_close_matches(pid, stems, n=1, cutoff=0.85)
            if matches:
                return self._stem_to_path[matches[0]]
        except Exception:
            pass
        
        return None
    
    def _build_valid_patient_list(self, patient_files: List[Path]) -> List[Path]:
        """Build list of patient files that have matching annotations."""
        
        csv_patient_ids = set(self.annotations_df['patient-id'].astype(str).str.strip())
        valid_files = []
        missing_ids = []
        
        print(f"\n🔍 Matching {len(patient_files)} files to {len(csv_patient_ids)} CSV entries...")
        
        for file_path in patient_files:
            # Handle .nii.gz double extension
            patient_id = file_path.stem  # Gets "DLCS_0001.nii" from "DLCS_0001.nii.gz"
            if patient_id.endswith('.nii'):
                patient_id = patient_id[:-4]  # Remove .nii to get "DLCS_0001"
            
            # Direct match
            if patient_id in csv_patient_ids:
                valid_files.append(file_path)
            else:
                missing_ids.append(patient_id)
        
        if len(missing_ids) > 0 and len(missing_ids) <= 10:
            print(f"⚠️ Warning: {len(missing_ids)} files have no CSV annotations:")
            for mid in missing_ids[:10]:
                print(f"   - {mid}")
        elif len(missing_ids) > 10:
            print(f"⚠️ Warning: {len(missing_ids)} files have no CSV annotations (first 10):")
            for mid in missing_ids[:10]:
                print(f"   - {mid}")
        
        if len(valid_files) == 0:
            print(f"\n❌ NO MATCHES FOUND!")
            print(f"File stems (first 5): {[f.stem for f in patient_files[:5]]}")
            print(f"CSV IDs (first 5): {sorted(list(csv_patient_ids))[:5]}")
            
            raise RuntimeError(
                f"No patient files could be matched to CSV annotations!\n"
                f"Files checked: {len(patient_files)}\n"
                f"CSV entries: {len(csv_patient_ids)}"
            )
        
        print(f"✓ Matched {len(valid_files)}/{len(patient_files)} files to CSV annotations")
        
        # Update mapping with clean IDs
        self._stem_to_path = {}
        for p in valid_files:
            clean_id = p.stem
            if clean_id.endswith('.nii'):
                clean_id = clean_id[:-4]
            self._stem_to_path[clean_id] = p
        
        return valid_files
    
    def __len__(self) -> int:
        return len(self.patient_files)
    
    def __getitem__(self, idx: int) -> Optional[Data]:
        patient_file = self.patient_files[idx]
        
        # Handle .nii.gz double extension
        patient_id = patient_file.stem
        if patient_id.endswith('.nii'):
            patient_id = patient_id[:-4]
        
        if patient_id in self.data_cache:
            return self.data_cache[patient_id]
        
        image_array, spacing, origin, direction, image_sitk = \
            self.preprocessor.load_nifti(str(patient_file))
        
        if image_array is None:
            return None
        
        image_normalized = self.preprocessor.normalize_hu(image_array)
        
        patient_annotations = self.annotations_df[
            self.annotations_df['patient-id'] == patient_id
        ]
        
        if len(patient_annotations) == 0:
            for csv_id in self.annotations_df['patient-id'].unique():
                if self._resolve_patient_file(csv_id) == patient_file:
                    patient_annotations = self.annotations_df[
                        self.annotations_df['patient-id'] == csv_id
                    ]
                    break
        
        if len(patient_annotations) == 0:
            return None
        
        nodule_coords = []
        nodule_features = []
        nodule_labels = []
        
        for _, row in patient_annotations.iterrows():
            try:
                # Use SimpleITK's built-in transform (handles all coordinate systems)
                try:
                    point_world = (float(row['coordX']), float(row['coordY']), float(row['coordZ']))
                    idx_voxel = image_sitk.TransformPhysicalPointToIndex(point_world)
                    
                    # Unpack (SimpleITK returns (x, y, z) in voxel space)
                    x_voxel, y_voxel, z_voxel = idx_voxel
                    
                    # Safety clamp to image bounds
                    x_voxel = max(0, min(x_voxel, image_array.shape[2] - 1))
                    y_voxel = max(0, min(y_voxel, image_array.shape[1] - 1))
                    z_voxel = max(0, min(z_voxel, image_array.shape[0] - 1))
                    
                except Exception as e:
                    # Fallback only if transform fails
                    print(f"  ⚠️ {patient_id}: Transform failed ({e}), skipping nodule")
                    continue

                # Create coordinate tuple (z, y, x) for numpy array indexing
                coord = (z_voxel, y_voxel, x_voxel)
                
                # Get nodule diameter to adjust patch size
                diameter_mm = float(row.get('w', 10.0))

                # Use 2x diameter as patch size (minimum 20mm, max 40mm)
                patch_size_mm = max(20, min(40, diameter_mm * 2))
                patch = self.preprocessor.extract_nodule_patch(image_normalized, coord, diameter_mm=patch_size_mm, spacing=spacing)
                mask = np.ones_like(patch, dtype=np.int32)
                radiomics_features = self.preprocessor.extract_radiomics_features(patch, mask)
                
                diameter = float(row.get('w', 0))
                spatial_features = np.array([z_voxel, y_voxel, x_voxel, diameter], dtype=np.float32)
                
                # FIX: Center crop instead of corner crop
                flat_len = patch.size
                mid_idx = flat_len // 2
                start_idx = max(0, mid_idx - 50)
                end_idx = start_idx + 100
                
                patch_flat = patch.flatten()[start_idx:end_idx]
                
                if len(patch_flat) < 100:
                    patch_flat = np.pad(patch_flat, (0, 100 - len(patch_flat)), mode='constant')
                
                full_features = np.concatenate([
                    spatial_features,
                    radiomics_features,
                    patch_flat
                ], dtype=np.float32)
                
                # Debug first nodule only
                if len(nodule_features) == 0:
                    rad_nz = np.count_nonzero(radiomics_features)
                    vis_nz = np.count_nonzero(patch_flat)
                    print(f"🔍 {patient_id}: Radiomics {rad_nz}/18, Visual {vis_nz}/100")
                    if rad_nz == 0 or vis_nz < 10:
                        print(f"  ⚠️ Poor features detected!")
                
                nodule_coords.append([z_voxel, y_voxel, x_voxel])
                nodule_features.append(full_features)
                nodule_labels.append(self._map_to_multiclass(row))
                
            except Exception as e:
                print(f"Error processing nodule in {patient_id}: {e}")
                continue
        
        if len(nodule_features) == 0:
            return None
        
        nodule_coords_array = np.array(nodule_coords, dtype=np.float32)
        nodule_features_array = np.array(nodule_features, dtype=np.float32)
        nodule_labels_array = np.array(nodule_labels, dtype=np.int64)
        
        # FIX: RobustScaler with safety checks
        from sklearn.preprocessing import RobustScaler
        
        scaler = RobustScaler()
        
        try:
            nodule_features_array = scaler.fit_transform(nodule_features_array)
            
            # Handle NaN from zero-variance features
            nan_mask = np.isnan(nodule_features_array)
            if nan_mask.any():
                for col in range(nodule_features_array.shape[1]):
                    if np.isnan(nodule_features_array[:, col]).any():
                        original_features = np.array(nodule_features)
                        nodule_features_array[:, col] = original_features[:, col]
            
            # Clip extremes
            nodule_features_array = np.clip(nodule_features_array, -10, 10)
            
        except Exception as e:
            print(f"⚠️ Scaling failed for {patient_id}, using fallback")
            # Fallback: simple normalization
            for col in range(nodule_features_array.shape[1]):
                col_min = nodule_features_array[:, col].min()
                col_max = nodule_features_array[:, col].max()
                if col_max - col_min > 1e-10:
                    nodule_features_array[:, col] = \
                        (nodule_features_array[:, col] - col_min) / (col_max - col_min)
        
        # Hypergraph construction
        try:
            hyperedges_dict = self.hypergraph_constructor.construct_hypergraph(
                nodule_coords_array, nodule_features_array
            )
            all_hyperedges = (
                hyperedges_dict['spatial'] +
                hyperedges_dict['feature'] +
                hyperedges_dict['anatomical']
            )
        except:
            all_hyperedges = []
        
        if len(all_hyperedges) == 0:
            if len(nodule_features_array) >= 2:
                all_hyperedges = [[i, j] for i in range(len(nodule_features_array))
                                for j in range(i + 1, len(nodule_features_array))]
            else:
                all_hyperedges = [[0]]
        
        num_nodes = len(nodule_features_array)
        edge_list = []
        
        for hyperedge in all_hyperedges:
            if len(hyperedge) == 1:
                edge_list.append([hyperedge[0], hyperedge[0]])
            else:
                for i in range(len(hyperedge)):
                    for j in range(i + 1, len(hyperedge)):
                        node_i = hyperedge[i]
                        node_j = hyperedge[j]
                        
                        if node_i < num_nodes and node_j < num_nodes:
                            edge_list.append([node_i, node_j])
                            edge_list.append([node_j, node_i])
        
        if len(edge_list) == 0:
            edge_list = [[0, 0]]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        data = Data(
            x=torch.tensor(nodule_features_array, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor([nodule_labels_array[0]], dtype=torch.long),
            patient_id=patient_id
        )
        
        # Edge validation
        if data.edge_index.numel() > 0:
            max_edge_idx = data.edge_index.max().item()
            if max_edge_idx >= num_nodes:
                valid_mask = (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)
                data.edge_index = data.edge_index[:, valid_mask]
                
                if data.edge_index.numel() == 0:
                    data.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        self.data_cache[patient_id] = data
        return data

    def _map_to_multiclass(self, row) -> int:
        """
        Binary classification: Benign (0) vs Malignant (1)
        """
        malignant = int(row.get('Malignant_lbl', 0))
        diameter = float(row.get('w', 0))
        
        # Class 1 if explicitly malignant OR large diameter
        if malignant == 1 or diameter >= 10.0:
            return 1
        else:
            return 0


def collate_hypergraph_batch(batch: List[Optional[Data]]) -> Optional[Batch]:
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return Batch.from_data_list(batch)