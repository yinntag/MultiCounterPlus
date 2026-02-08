import torch


def rearrange_features(features, batch_indices_object, frames_per_object):
    new_features = []
    start_idx = 0

    for num_objects in batch_indices_object:
        end_idx = start_idx + num_objects * frames_per_object

        batch_features = features[start_idx:end_idx]
        start_idx = end_idx

        for obj_idx in range(num_objects):
            indices = batch_features[obj_idx::num_objects]
            new_features.append(indices)

    new_features = torch.stack(new_features, dim=0).reshape(-1, features.size(1))
    return new_features


def restore_original_order(rearranged_features, batch_indices_object, frames_per_object):
    restored_features = []
    start_idx = 0

    for num_objects in batch_indices_object:
        end_idx = start_idx + num_objects * frames_per_object

        batch_features = rearranged_features[start_idx:end_idx]
        start_idx = end_idx

        for frame_idx in range(frames_per_object):
            indices = batch_features[frame_idx::frames_per_object]
            restored_features.append(indices)

    restored_features = torch.cat(restored_features, dim=0).reshape(-1, features.size(1))
    return restored_features