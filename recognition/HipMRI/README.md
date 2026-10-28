# Prostate MRI Segmentation using 3D Improved U-Net
- Author: William Armstrong (48842613)
- Course: COMP3710 - Pattern Analysis

---
title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph, and a figure/visualisation.


Background of the problem. Description of the model and the problem it solves. How it works in a paragraph and a figure/ visualisation.



---
## Feasibility Report

### User Need, Scope & Acceptance Criteria
The intended users of this segmentation model are imaging specialists. Their main needs are to speed up their work by moving from creating the anatomical contours to reviewing and adjusting the output of the model, while maintaining the safety of their patients. Therefore the purpose of the prototype is to be able to identify the different parts of the MRI and segment the images, ensuring to also reduce the under and over contouring of the prostate segmentation as that can be very dangerous to their patients.

To evaluate how well the model has addressed the problem some testable acceptance criteria have been made. For the 3D improved UNet to be worth it, it will further need to fix the issues around boundary and slice-transition errors and do it better than the 2D-UNet.

Testable acceptance criteria
1. Sorensen-Dice similarity greater than 0.7 on test set
2. Base and Apex specific Dice similarity greater than 0.7 on the test set
3. Intersection-over-Union (IoU) greater than 0.7 on test set
4. Less than 5 hours of training time and less than 20GB of peak VRAM on an a100 GPU
5. Noticeable difference between 2D and 3D models (> 5% improvement in test results)



### Model Choice & Course Concepts
The chosen dataset is HipMRI, focusing on Prostate Radiotherapy Planning. The chosen model architecture is the 3D Improved U-Net (Hard Difficulty). This model will be compared against a baseline 2D U-Net. This comparison will be used to weigh up if the larger computational expenses in the 3D improved U-Net will result in a better solution to the given problem. Testing whether it can improve upon the 2D U-Net's inability to take into account changes in 3D space (boundary and slice-transition errors) and whether this potential improvement is worth the increased compute required. The course concepts and literature support the choice of a 3D U-Net as U-Net's are used for segmenting data in high resolution using skip connections to avoid loosing context from the downsampling process. Further literature supports giving the model 3D context over 2D slices as a way of allowing it to understand the way the data reacts across slices which is the exact problem that needs to be fixed.

### Preliminary Feasibility Evidence

An initial data audit showed that there were a total of 211 scans, relating to 38 unique patients and each scan was a volume of shape 256x256x128. Further, it showed that slices 67-81 contained every class, moreover the prostate was present in a range from slices 45-81. A baseline smoke test on an A100 GPU for the 3D Improved U-Net showed that the data loaded correctly, split into train / validate / test, was able to run forward passes, train forward steps, evaluate and save checkpoints of the model. Further this smoke test allowed for measured resource estimation, with the dataset setup time taking 74 seconds, and each epoch taking approximately 35 seconds, the peak VRAM was also measured to be 21.46 GB. Estimating for approximately 50 epochs of training it would take approximately 30 minutes to train.


### Risks, Budget & Fallback
The key technical risks of this task are the memorisation of data (overfitting), lack of 3D data, cross contamination of patients data (not keeping the splits separate), high memory consumption, and accurate validation techniques.

The computing budget consists of usage of 100 tokens ($15) for an A100 GPU on Google Colab at an approximate usage of 6 tokens per hour allows for a maximum of 16 hours of use. This accounts for the training for the baseline, smoke test and full runs. This should be plenty of compute power and time to account for estimated resource consumption.

The Next planned experiment is a full training run against the data with a larger number of Epochs, then testing the accuracy of the boundaries and slice transitions in comparison to the 2D U-Net.

If the 3D Improved U-Net fails to train, a workable fallback would either be the 3D U-Net or the baseline 2D U-Net.




---
• Any dependencies required, including exact versions, and address reproducibility of results.







---
• Example inputs, outputs, and visual plots of your algorithm.






---
• Description of any specific pre-processing used, with references if any. Justify your training, validation, and testing splits of the data.






---
• Investigation of the Open Research Dilemma:
– Quantitative benchmarking of your chosen model against your implemented baseline model under
identical evaluation splits.
– Resource profiling table documenting computational efficiency (peak GPU VRAM, parameter
count, and inference runtime/latency).
– In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset
(diagnosing root triggers and failure modes).
– Actionable engineering recommendation and trade-off synthesis for your project manager.






---
• The mandatory Artificial Intelligence Usage Disclosure subsection (see Section 6).



