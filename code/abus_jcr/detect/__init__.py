"""Detectors sub-package (Phase 2+).

Houses the torchvision RetinaNet 2.5D detector, the frozen common per-slice
detection schema, the ``[2.0]`` Train design-constant probe, the detection
dataset/augmentation, training, inference, and cost instrumentation.

Import discipline: the correctness-critical, torch-free modules (:mod:`schema`,
:mod:`det_stats`, :mod:`augment_ops`, and the pure helpers in
:mod:`slice_det_dataset`) never import torch at module load, so they run in the
laptop's torch-free env. Torch/torchvision are imported lazily inside the
functions that need them (:mod:`retinanet`, :mod:`train`, :mod:`infer`,
:mod:`cost`), so importing this package does not require torch.
"""
