"""Paper-artifact rendering (tables, figures, artifact index) from saved analysis outputs.

Rendering only. The modules here read machine artifacts written by the
experiment code and format them; they never fit a model, run a test or change
an analysis input. Nothing in this package is imported by the per-image
pipeline, the segmenters, U-Net training or Experiment C, so adding or editing
it leaves those code fingerprints unchanged.
"""
