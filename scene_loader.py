"""Scene data loader from whichever Kairos stack is installed.

The ROS 2 stack ships the loader with kairos_python; the ROS 1 stack with
hydra_python.
"""
try:
    from kairos_python.data_loader import DataLoaderIter, get_dataloader
except ImportError:
    from hydra_python.data_loader import DataLoaderIter, get_dataloader

__all__ = ["DataLoaderIter", "get_dataloader"]
