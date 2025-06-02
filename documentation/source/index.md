# VibTracker: A Node-based framework for camera-based vibration measurement with DIC integration

Welcome to VibTracker, an innovative, open-source software designed to democratize image-based vibration measurement. Whether you're working in civil engineering, biomechanics, or material science, VibTracker offers flexibility to tailor your analysis workflows to specific experimental setups.

## VibTracker Features:

VibraTrack provides a modular, user-friendly interface based on the NodeEditor programmed by *Pavel Křupala*, [link to "node editor"](https://gitlab.com/pavel.krupala/pyqt-node-editor)

![Alt Text](./images/VibTracker_example.gif)

1. Customizable your post-process pipelines with node editor.
   - Create workflows based on your experimental setup.
   - Adapt and expand the software for various research and industrial applications.

2. Comprehensive functionality for camera-based vibration measurement
   - Calibration: Camera intrinsic / extrinsic calibration + distortion correction
   - Target Initialization: Define regions of interest with ease.
   - Tracking: Robust algorithms for target tracking.    
   - Displacement Calculation: Compute both 2D and 3D displacements from a variety of camera configurations.

3. Advanced capability : Digital Image Correlation (2D / 3D DIC)
   - Integrates DIC, a method for full-field displacement and strain measurement through speckle pattern images.
   - Supports single-camera, stereo-camera setup.

### Installation

To install VibTracker, visit the [Installation page](installation.md)

### Tutorials

To see the tutorials, visit the [Tutorials pages](tutorials.md)

```{toctree}
:maxdepth: 2
:caption: Contents:

installation.md
tutorials.md
```


