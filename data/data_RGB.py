from data.dataset_RGB import *
from data.dataset_blade import DataLoaderBladeFlowTrain, DataLoaderBladeFlowVal


def get_training_data(rgb_dir, meta, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderFileTrain(rgb_dir, meta, img_options)

def get_validation_data(rgb_dir, meta, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderFileVal(rgb_dir, meta, img_options)

def get_test_data(meta, input_dir, target_dir, img_options):
    assert os.path.exists(meta)
    assert os.path.exists(input_dir)
    assert os.path.exists(target_dir)
    return DataLoaderFileTest(meta, input_dir, target_dir, img_options)


def get_blade_training_data(rgb_dir, meta, img_options, flow_dir=None):
    """Return a blade dataset loader that optionally includes optical flow labels.

    Args:
        rgb_dir:    Root directory of the dataset (sharp/ and blur/ sub-dirs).
        meta:       Meta list file (``sharp_rel blur_rel`` per line).
        img_options: Dict with at least ``patch_size``.
        flow_dir:   Optional directory containing pre-computed flow .npy files.
                    If None, flows are computed on-the-fly (slower).

    Returns:
        DataLoaderBladeFlowTrain instance.
    """
    assert os.path.exists(rgb_dir)
    return DataLoaderBladeFlowTrain(rgb_dir, meta, img_options, flow_dir=flow_dir)


def get_blade_validation_data(rgb_dir, meta, img_options, flow_dir=None):
    """Return a blade validation dataset loader."""
    assert os.path.exists(rgb_dir)
    return DataLoaderBladeFlowVal(rgb_dir, meta, img_options, flow_dir=flow_dir)
