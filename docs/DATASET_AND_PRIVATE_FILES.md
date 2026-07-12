# Dataset and Private Files

This release excludes:

- FaceScape raw meshes, scans, textures, and rendered portraits
- FaceScape-derived mesh or image files
- HRN weights and HRN output mesh caches
- OBJ, PLY, STL, NPY, NPZ, PKL, PT, PTH, image, and archive files
- Private server utilities and local machine configuration
- Local quick-test data and figure asset packages
- Exploratory experiment archives that are not used by the manuscript tables

External resources must be obtained separately from their official providers:

- FaceScape dataset: <https://nju-3dv.github.io/projects/FaceScape/>
- FaceScape license agreement: <https://facescape.nju.edu.cn/static/License_Agreement.pdf>
- HRN official implementation: <https://github.com/younglbw/hrn>
- HRN project page: <https://younglbw.github.io/HRN-homepage/>

Prepare the dataset locally and set `PAPERFIT_ROOT` before rerunning experiments. Do not commit licensed FaceScape data, FaceScape-derived assets, HRN weights, or HRN output meshes to this repository.

