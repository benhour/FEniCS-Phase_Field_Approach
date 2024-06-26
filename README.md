# FEniCS-Phase_Field_Approach
In this project, a theoretical-computational framework is proposed for predicting the failure behavior of two anisotropic brittle materials, namely, single crystal magnesium and boron carbide under quasi-static and dynamic loading. Constitutive equations are derived, in both small and large deformations, by using thermodynamics in order to establish a fully coupled and transient twin and crack system. To study the common deformation mechanisms (e.g., twinning and fracture), a monolithically-solved Ginzburg–Landau-based phase-field theory coupled with the mechanical equilibrium equation is implemented in a finite element simulation framework for the following problems: (i) twin evolution in two-dimensional single crystal magnesium and boron carbide under simple shear deformation; (ii) crack-induced twinning for magnesium under pure mode I and mode II loading; and (iii) study of fracture in homogeneous single crystal boron carbide under biaxial compressive loading.



## FEniCS
An open-source computing platform, [FEniCS](https://fenicsproject.org/) is used for translating the governing equations into efficient finite element code. An Intel Xeon E7-4850 (in total 64 cores each 40 MB cache, equipped with 256 GB RAM in total, running Linux Kernel 5 Ubuntu 20.04) is implemented for running the simulations. Running the codes in parallel requires the following command:
```
mpirun -n (# of CPUs) python3 script_test.py
```



## SALOME
The geometry and meshes are created in [SALOME Version 9.7.0](https://www.salome-platform.org/?page_id=15) as MED files and then converted to XML files for the simulations using **DOLFIN** package.



## Citation
If you find these codes useful, please cite our open access works as [^1][^2][^3]:
```
@article{amirian2023study,
 title={The study of diffuse interface propagation of dynamic failure in advanced ceramics using the phase-field approach},
 author={Amirian, Benhour and Abali, Bilen Emek and Hogan, James David},
 journal={Computer Methods in Applied Mechanics and Engineering},
 volume={405},
 pages={115862},
 year={2023},
 publisher={Elsevier}
}


@article{AMIRIAN2022111789,
title = {Thermodynamically-consistent derivation and computation of twinning and fracture in brittle materials by means of phase-field approaches in the finite element method},
author = {Benhour Amirian and Hossein Jafarzadeh and Bilen Emek Abali and Alessandro Reali and James David Hogan}
journal = {International Journal of Solids and Structures},
volume = {252},
issn = {0020-7683},
pages = {111789},
year = {2022},
publisher = {Elsevier}
}


@article{amirian2022phase,
 title={Phase-field approach to evolution and interaction of twins in single crystal magnesium},
 author={Amirian, Benhour and Jafarzadeh, Hossein and Abali, Bilen Emek and Reali, Alessandro and Hogan, James David},
 journal={Computational Mechanics},
 volume={70},
 number={4},
 pages={803--818},
 year={2022},
 publisher={Springer}
}
```
[^1]: [Computer Methods in Applied Mechanics and Engineering](https://doi.org/10.1016/j.cma.2022.115862)
[^2]: [International Journal of Solids and Structures](https://doi.org/10.1016/j.ijsolstr.2022.111789)
[^3]: [Computational Mechanics](https://doi.org/10.1007/s00466-022-02209-3)


[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
