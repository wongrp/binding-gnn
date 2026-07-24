"""Backward compatibility: TPConvLite is now TPConv with tp_mode='diagonal'."""

from src.layers.tpconv import TPConv as TPConvLite  # noqa: F401

__all__ = ["TPConvLite"]
