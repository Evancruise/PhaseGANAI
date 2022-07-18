# PhaseGANAI

This project aims to develop an automatic recognizing image system for early detection of breast cancer.

## Conventional imaging technique
Conventional imaging technique is based on the contrast of the attenuation ma­terial within an object. It is proven that it can reach great vi­sualization in high X­-ray energy and low radiation dose. However, the image quality is refrained and only valid for weak absorbing objects.  By contrast, current imaging technique, phase contrast imaging, emerges with improved phase sensitivity whose magnitude in soft tissue is 3 orders higher than the attenuation part especially in the presence of tissues with similar density. PCI techniques can enhance the contrast and sensitivity by exploiting the phase contrast. It has seen a large crease in interest and development since 25 years ago. With the use of phase retrieval algorithms together with CT, the complex refractive index distribution in the imaged object can be reconstructed in 3D.
Phase retrieval algorithms are realized base on the linearization of the Fresnel integral (the solutions are not applicable for all conditions (partially followed))
Challenges:
Relatively short distances (paraxial approximation): edge-enhancement regime (assumptions: homogeneity of constituent materials, but not apt to various types of samples).
Far distances: enhanced visible diffraction fringe.

## Phase contrast imaging
Phase contrast imaging concentrates on the phase shift in the phase of X-ray beam that passing through the object to create X-ray images. It mainly depends on a decrease of the X-ray beam’s intensity, which can be directly measured with the assistance of X-ray detector. Its advantage is that it is  more sensitive to density variations in the sample than the conventional imaging (attenuation imaging), yielding an improved soft tissue contrast. Theoretically, phase-contrast X-ray imaging has higher contrast and sensitivity thanconventional absorption imaging. Higher contrast makes the different compositions of anobject more distinguishable.

## Phase retrieval algorithm (stated in phase_retrieval.m)
Phase retrieval is the process of algorithmically finding solutions to the phase problem, which can be seved as a kind of nonlinear inverse problem. The major obstacle in the phase retrieval problem is the nonlinear relationship between the intensity (amplitude) and phase of the sample in the image formation process. Although this nonlinear inverse problem can be solved by the iterative phase retrieval algorithms, they rely on computationally intensive iterative operations, and there is no theoretical convergence guarantee, therefore the image degradation will be occurred.

## Steps for image statistical analytics

1. Set the parameters prepared for generating X-ray breast mammography images
command:
```
Step0_DefineParameters_Script.m
```
2. Produce these images using CLB (Clustered Lumpy Background) models
3. Navigate forward propagation to acquire X-ray images on digital detectors
4. Apply phase retrieval algorithms to reconstruct images we actually get
'''
Step1_GenerateRealizations.m
'''
5. Use statistical observers to analyze image quality and detactability of signals (Gaussian sphere object, represented as some precancerous areas).
```
Step2_CalculateCovofRealizations
Step3_template_observer_HO.m
```

## Deep learning algorithms

The area of the signal is overlapped with that of high-density tissues in some phase images, which defers the process of image restoration. As a result, the relationship between the object plane and detector planes cannot be well described. Deep learning techniques have been used to establish a non-linear mapping relationship between the input data and the reconstructed data within the regime of phase-contrast imaging. Here, GAN structure based on the iterative approach is applied to deal with the aforementioned issues. In this part, you can browse the files in 'step04_phaseGAN_part' folder.

## Citation

```
@article{guigay2007mixed,
  title={Mixed transfer function and transport of intensity approach for phase retrieval in the Fresnel region},
  author={Guigay, Jean Pierre and Langer, Max and Boistel, Renaud and Cloetens, Peter},
  journal={Optics letters},
  volume={32},
  number={12},
  pages={1617--1619},
  year={2007},
  publisher={Optical Society of America}
}

@article{langer2008quantitative,
  title={Quantitative comparison of direct phase retrieval algorithms in in-line phase tomography},
  author={Langer, Max and Cloetens, Peter and Guigay, Jean-Pierre and Peyrin, Fran{\c{c}}oise},
  journal={Medical physics},
  volume={35},
  number={10},
  pages={4556--4566},
  year={2008},
  publisher={Wiley Online Library}
}
```


