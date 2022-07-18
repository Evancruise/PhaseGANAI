# PhaseGANAI

This project aims to develop an automatic recognizing image system for early detection of breast cancer.

# Conventional imaging technique
Conventional imaging technique is based on the contrast of the attenuation ma­terial within an object. It is proven that it can reach great vi­sualization in high X­-ray energy and low radiation dose. However, the image quality is refrained and only valid for weak absorbing objects.  By contrast, current imaging technique, phase contrast imaging, emerges with improved phase sensitivity whose magnitude in soft tissue is 3 orders higher than the attenuation part especially in the presence of tissues with similar density. PCI techniques can enhance the contrast and sensitivity by exploiting the phase contrast. It has seen a large crease in interest and development since 25 years ago. With the use of phase retrieval algorithms together with CT, the complex refractive index distribution in the imaged object can be reconstructed in 3D.
Phase retrieval algorithms are realized base on the linearization of the Fresnel integral (the solutions are not applicable for all conditions (partially followed))
Challenges:
Relatively short distances (paraxial approximation): edge-enhancement regime (assumptions: homogeneity of constituent materials, but not apt to various types of samples).
Far distances: enhanced visible diffraction fringe.

# Phase contrast imaging
Phase contrast imaging concentrates on the phase shift in the phase of X-ray beam that passing through the object to create X-ray images. It mainly depends on a decrease of the X-ray beam’s intensity, which can be directly measured with the assistance of X-ray detector. Its advantage is that it is  more sensitive to density variations in the sample than the conventional imaging (attenuation imaging), yielding an improved soft tissue contrast. Theoretically, phase-contrast X-ray imaging has higher contrast and sensitivity thanconventional absorption imaging. Higher contrast makes the different compositions of anobject more distinguishable.

# Phase retrieval algorithm (stated in phase_retrieval.m)
Phase retrieval is the process of algorithmically finding solutions to the phase problem, which can be seved as a kind of nonlinear inverse problem. The major obstacle in the phase retrieval problem is the nonlinear relationship between the intensity (amplitude) and phase of the sample in the image formation process. Although this nonlinear inverse problem can be solved by the iterative phase retrieval algorithms, they rely on computationally intensive iterative operations, and there is no theoretical convergence guarantee, therefore the image degradation will be occurred.




