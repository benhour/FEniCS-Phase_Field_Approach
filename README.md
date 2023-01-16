# FEniCS-Phase_Field_Approach
A phase field approach for twinning and fracture in brittle materials using monolithic scheme
A theoretical-computational framework is proposed for predicting the failure behavior of two anisotropic brittle materials, namely, single crystal magnesium and boron carbide. Constitutive equations are derived, in both small and large deformations, by using thermodynamics in order to establish a fully coupled and transient twin and crack system. To study the common deformation mechanisms (e.g., twinning and fracture), which can be caused by extreme mechanical loading, a monolithically-solved Ginzburg–Landau-based phase-field theory coupled with the mechanical equilibrium equation is implemented in a finite element simulation framework for the following problems: (i) twin evolution in two-dimensional single crystal magnesium and boron carbide under simple shear deformation; (ii) crack-induced twinning for magnesium under pure mode I and mode II loading; and (iii) study of fracture in homogeneous single crystal boron carbide under biaxial compressive loading. The results are verified by a steady-state phase-field approach and validated by available experimental data in the literature. The success of this computational method relies on using two distinct phase-field (order) parameters related to fracture and twinning. A finite element method-based code is developed within the Python-based open-source platform FEniCS. We make the code publicly available and the developed algorithm may be extended for the study of phase transformations under dynamic loading or thermally-activated mechanisms, where the competition between various deformation mechanisms is accounted for within the current comprehensive modeling approach.
