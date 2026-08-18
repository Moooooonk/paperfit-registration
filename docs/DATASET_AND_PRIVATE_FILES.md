# Dataset and Private Files

This release excludes:

- FaceScape raw meshes, scans, textures, and rendered portraits
- FaceScape-derived mesh or image files
- HRN weights and HRN output mesh caches
- 3DDFA-V2 weights and 3DDFA-V2 output mesh caches
- Blinded perceptual-rating panels and private case keys
- OBJ, PLY, STL, NPY, NPZ, PKL, PT, PTH, image, and archive files
- Private server utilities and local machine configuration
- Local quick-test data and figure asset packages
- Exploratory experiment archives that are not used by the manuscript tables

External resources must be obtained separately from their official providers:

- FaceScape dataset: <https://nju-3dv.github.io/projects/FaceScape/>
- FaceScape license agreement: <https://nju-3dv.github.io/projects/FaceScape/static/license/LicenseAgreement_FaceScape.pdf>
- HRN official implementation: <https://github.com/younglbw/hrn>
- HRN project page: <https://younglbw.github.io/HRN-homepage/>
- 3DDFA-V2 official implementation: <https://github.com/cleardusk/3DDFA_V2>

Prepare the dataset locally and set `PAPERFIT_ROOT` before rerunning
experiments. Do not commit licensed FaceScape data, FaceScape-derived assets,
rating panels, private case keys, reconstruction weights, or reconstruction
output meshes to this repository.

