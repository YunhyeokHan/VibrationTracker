# VibrationTracker: A Node-based framework for camera-based vibration measurement with DIC integration

Welcome to VibrationTracker, an innovative, open-source software designed to democratize image-based vibration measurement. Whether you're working in civil engineering, biomechanics, or material science, VibTracker offers flexibility to tailor your analysis workflows to specific experimental setups.

## VibrationTracker Features:

VibrationTracker provides a modular, user-friendly interface based on the NodeEditor programmed by *Pavel Křupala*, [link to "node editor"](https://gitlab.com/pavel.krupala/pyqt-node-editor)

![Alt Text](./documentation/source/images/VibTracker_example.gif)

1. Customizable post-process pipelines with node editor:
   - Create workflows based on your experimental setup.
   - Adapt and expand the software for various research and industrial applications.

2. Comprehensive functionality for camera-based vibration measurements
   - Calibration: camera intrinsic / extrinsic calibration + distortion correction
   - Target Initialization: define regions of interest with ease.
   - Tracking: robust algorithms for target tracking.    
   - Displacement Calculation: compute both 2D and 3D displacements from a variety of camera configurations.

3. Advanced vibration tracking capability by Digital Image Correlation (2D / 3D DIC):
   - Integrates DIC, a method for full-field displacement and strain measurement through speckle pattern images.
   - Supports single-camera, stereo-camera setup.

### Installation

To install VibrationTracker, visit the [Installation page](./documentation/source/installation.md)

### Tutorials

To see the VibrationTracker, visit the [Tutorials pages](./documentation/source/tutorials.md)


### Acknowledgement:

This software was initiated during [project Dynatimbereyes](https://anr.fr/Projet-ANR-21-CE22-0027) funded by the french ANR (ANR-21-CE22-0027), in collaboration with  [Stefania Lo Feudo](https://www.researchgate.net/profile/Stefania-Lo-Feudo), [Gwendal Cumunel](https://www.researchgate.net/profile/Gwendal-Cumunel), and [Franck Renaud](https://www.researchgate.net/profile/Franck-Renaud) 

### Citation: 

``` 
@PHDTHESIS{Han2024,
url = "2024UPAST190",
title = "Innovative video-based techniques for outdoor vibration analysis of timber buildings
",
author = "Han, Yunhyeok",
year = "2024",
note = "Thèse de doctorat dirigée par Renaud, Franck" et encadrée par Lo Feudo, Stefania et Cumunel, Gwendal.
note =  "Mécanique des solides et des structures université Paris-Saclay 2024",
note = "s339344",
url = "theses.fr/s339344",
}
```

DOI ZENODO


### Use cases....

1. Lo Feudo, S., Han, Y., Cumunel, G., Hoult, R., Bertholet, A., Garnier, D., Candeias, P., Correia, A.A. and Pacheco de Almeida, J. (2025), Video Tracking of Targets for Vibration Measurement of Large-Scale Structures Under Seismic Excitation. Earthquake Engng Struct Dyn., 54: 2106-2120. https://doi.org/10.1002/eqe.4353