# Prostate MRI Segmentation using 3D Improved U-Net
- Author: William Armstrong (48842613)

---
### Background & Problem
In Prostate Radiotherapy Planning (HipMRI), targeting a beam requires millimeter precision between the prostate and the surrounding critical organs (bladder, rectum, femoral heads), under and over contouring is not viable either as under contouring risks recurrence of tumours and over contouring irradiates the surrounding organs causing massiveissues. There are two main issues when trying to make models for this data. Firstly, as it is 3D data, if a 2D model attempts to learn it through slices it may result in jagged 3D boundaries and secondly, the prostate is rather small, with it making up less than 2% of the overall 3D volume, this requires thought on how best to score losses.

### Adressing the Problem

To address these issues, this project implements a 3D improved U-Net (Isensee et al, 2018). It allows the model to process whole volumes with 3D convolutions meaning that every prediction is informed by the slices above and below it. It is trained with a multi-class Dice loss that weights every class equally regardless of the size. The model will be analysed against a 2D U-Net baseline on the exact same patient level split in order to discover the effect 3D context has and whether it is capable of removing boundary and slice-transition errors or whether it just improves the easy central slices.

![Model Diagram](model_diagram.drawio.png)
Figure 1: Model Architecture Flow

### How it works

The network in Figure 1 is an 3D Improved U-Net, this means that it is an starts with an encoder, then a decoder and has skip connections.

The Encoder (downsampling, blue) has five levels with 16, 32, 64, 128 and 256 channels. Each level starts with a 3x3x3 convolution with stride 2 (except level 1 which uses stride 1) in order to halve the spatial dimension. This brings a 256x256x128 down to the bottleneck of 16x16x8.

The output from each encoder is then passed into a residual block that is made up of two 3x3x3 convolutions, each preceeded by instance normalisation, leaky ReLU (slope = 0.01) and dropout (p=0.3). The output from the residual block is then added to the original output from the encoder to make that levels skip connection (dotted arrows).

The model then upscales by a factor of two using the nearest neighbour, then uses a 3x3x3 convolution to halve the channels, the result is concatenated with the matching skip connection which is used to recover the spatial detail lost during the downsampling. Then a localisation module merges the two and ensures that the channels halve.

Lastly, the deep supervision is made up of three 1x1x1 convolutions that produce six class segmentation maps at levels 3, 2 and 1. Each coarser map is trilinearly upsampled and added element-wise to the next. This means the final output combines predictions from three different levels and allows for the gradients to directly affect deeper layers.


---
## Feasibility Report

### User Need, Scope & Acceptance Criteria
The intended users of this segmentation model are imaging specialists. Their main needs are to speed up their work by moving from creating the anatomical contours to reviewing and adjusting the output of the model, while maintaining the safety of their patients. Therefore the purpose of the prototype is to be able to identify the different parts of the MRI and segment the images, ensuring to also reduce the under and over contouring of the prostate segmentation as that can be very dangerous to their patients.

To evaluate how well the model has addressed the problem some testable acceptance criteria have been made. For the 3D improved UNet to be worth it, it will further need to fix the issues around boundary and slice-transition errors and do it better than the 2D-UNet.

Testable acceptance criteria:
1. Sorensen-Dice similarity greater than 0.7 on test set
2. Base and Apex specific Dice similarity greater than 0.7 on the test set
3. Intersection-over-Union (IoU) greater than 0.7 on test set
4. Less than 5 hours of training time and less than 20GB of peak VRAM on an a100 GPU
5. Noticeable difference between 2D and 3D models (> 5% improvement in test results)



### Model Choice & Course Concepts
The chosen dataset is HipMRI, focusing on Prostate Radiotherapy Planning. The chosen model architecture is the 3D Improved U-Net (Hard Difficulty). This model will be compared against a baseline 2D U-Net. This comparison will be used to weigh up if the larger computational expenses in the 3D improved U-Net will result in a better solution to the given problem. Testing whether it can improve upon the 2D U-Net's inability to take into account changes in 3D space (boundary and slice-transition errors) and whether this potential improvement is worth the increased compute required. The course concepts and literature support the choice of a 3D U-Net as U-Net's are used for segmenting data in high resolution using skip connections to avoid loosing context from the downsampling process. Further literature supports giving the model 3D context over 2D slices as a way of allowing it to understand the way the data reacts across slices which is the exact problem that needs to be fixed.

### Preliminary Feasibility Evidence

An initial data audit showed that there were a total of 211 scans, relating to 38 unique patients and each scan was a volume of shape 256x256x128. The data was split into 'semantic_MRs' containing the raw MRI volumes and 'semantic_labels_only' containing the source of truth for the segmentation by label. Further, it showed that slices 67-81 contained every class, moreover the prostate was present in a range from slices 45-81. A baseline smoke test on an A100 GPU for the 3D Improved U-Net showed that the data loaded correctly, split into train / validate / test, was able to run forward passes, train forward steps, evaluate and save checkpoints of the model. Further this smoke test allowed for measured resource estimation, with the dataset setup time taking 74 seconds, and each epoch taking approximately 35 seconds, the peak VRAM was also measured to be 21.46 GB. Estimating for approximately 50 epochs of training it would take approximately 30 minutes to train.


### Risks, Budget & Fallback
The key technical risks of this task are the memorisation of data (overfitting), lack of 3D data, cross contamination of patients data (not keeping the splits separate), high memory consumption, and accurate validation techniques.

The computing budget consists of usage of 100 tokens ($15) for an A100 GPU on Google Colab at an approximate usage of 6 tokens per hour allows for a maximum of 16 hours of use. This accounts for the training for the baseline, smoke test and full runs. This should be plenty of compute power and time to account for estimated resource consumption.

The Next planned experiment is a full training run against the data with a larger number of Epochs, then testing the accuracy of the boundaries and slice transitions in comparison to the 2D U-Net. If the 3D Improved U-Net fails to train, a workable fallback would either be the 3D U-Net or the baseline 2D U-Net.

---
## Run Instructions
Ensure that the following depencencies are installed in the environment that this project is being run.

Set the dataset root path in train.py to the path containing the semantic_MRs and semantic_labels_only folders. Further it is optional to adjust the hyperparameters in train.py or the random seed in dataset.py to adjust the train / validate / test split. Further although the hyperparameters are the same and the data split is the same it does not necessarily mean that the results will be repeated exactly due to random features such as the dropout or many other factors.

### Dependencies
- Python 3.12.3
- torch 2.14.1
- torchvision 0.29.1
- numpy 2.5.4
- nibabel 5.4.2
- matplotlib 3.11.2
- tqdm 4.70.1

---
## Inputs, Outputs, and Plots
### 3D Improved U-Net
The type of inputs that are shown in the model can be seen bellow in Figure 2. As the model takes 3D data (256x256x128), it can be thought of as if 128 of the 2D input images are stacked ontop of eachother and passed into the model. Further the ground truth is also taken as an input into the model, although it is used purely for generating a loss function and evaluating how well the model predicted it.

![Model Input, Output and Truth img 0](3D_Model/predictions/unet3d_prediction_0.png)
Image 1
![Model Input, Output and Truth img 1](3D_Model/predictions/unet3d_prediction_1.png)
Image 2
![Model Input, Output and Truth img 2](3D_Model/predictions/unet3d_prediction_2.png)
Image 3
Figure 2: Model input, label ground truth, prediction segmentation (Dark purple: background, Blue purple: body, Teal: bone, Green: bladder, light green: rectum, yellow: prostate)

The model ran for 57 minutes and had a peak VRAM of 21.47GB. The outputs of the model are both as seen above and also a json file containing information of the data (due to the fact some environments do not have matplotlib). The information from this json file has been used to plot the following, looking at the dice score and the loss in validation and test per epoch.

![Dice per class per epoch](3D_Model/unet3d_dice.png)
Figure 3: Dice per class per Epoch

![Loss per epoch of training and validation](3D_Model/unet3d_loss.png)
Figure 4: Training and Validation Loss of each epoch 

Figure 3 shows that the per class DSC was much climbed very fast for the first 20 epochs and then slowly improved as it trained further epochs. Figure 4 shows the train and validation loss tightly linked together the whole way down with the validation loss just a bit behind the train loss. This closeness of the losses shows that the model did not fall into a trap of memorising data.

Globally the 3D model achieves a prostate Dice Similarity Coefficient (DSC) of 0.86 and an Intersection over Union (IoU) of 0.7562 which exceeds the 0.7 target. However, this score masks a weakness over the five most superior (base) slices obtaining a DSC of 0.7629 and an IoU of 0.6275.

```
Test DSC Background: 0.9978
Test DSC Body: 0.9880
Test DSC Bone: 0.9267
Test DSC Bladder: 0.9591
Test DSC Rectum: 0.8821
Test DSC Prostate: 0.8601
Min class DSC: 0.8601 | Mean DSC: 0.9356

Prostate by region (5 slices at each end):
whole: DSC 0.8601 | IoU 0.7562
 apex: DSC 0.8419 | IoU 0.7455
 base: DSC 0.7629 | IoU 0.6275
```


### 2D U-Net

![Model Input, Output and Truth img 0](2D_Unet_output/predictions/unet_2d_prediction_0.png)
Image 1
![Model Input, Output and Truth img 1](2D_Unet_output/predictions/unet_2d_prediction_1.png)
Image 2
![Model Input, Output and Truth img 2](2D_Unet_output/predictions/unet_2d_prediction_2.png)
Image 3
Figure 5: Model input, label ground truth, prediction segmentation (Dark purple: background, Blue purple: body, Teal: bone, Green: bladder, light green: rectum, yellow: prostate)

The model ran for 55 minutes and had a peak VRAM of 20.54GB. The outputs of the model are both as seen above and also a json file containing information of the data (due to the fact some environments do not have matplotlib). The information from this json file has been used to plot the following, looking at the dice score and the loss in validation and test per epoch.


![Dice per class per epoch](2D_Unet_output/unet_2d_dice.png)
Figure 6: Dice per class per Epoch

![Loss per epoch of training and validation](2D_Unet_output/unet_2d_loss.png)
Figure 7: Training and Validation Loss of each epoch 


Figure 6 shows that the per class DSC climbed very fast initially (first 20 epochs) but then mostly leveled off for the rest of the training. Figure 7 shows that similar to figure 4 there was no memorisation of data and the validation loss stayed tightly linked to the training loss, trailing just a bit behind.


---
• Description of any specific pre-processing used, with references if any. Justify your training, validation, and testing splits of the data.

### Pre-Processing Techniques
Pre-processing was used in order to avoid data memorisation, data leaking and ensure that the model was good generally, not just on the training set.

Firstly, each MRI is paired up with its corresponding label using the `get_pairs` function, this is important as a mismatched pair would evaluate the model on the wrong thing. 





### Data Split of Train, Validation, Test
To ensure that there was no data leaking there was a patient level split, not just a split based on week, volume, or slice. `split_pairs` groups by `Case_AAA` so that all a patients weekly scans are in one place. This is crucial as with 211 volumes and 38 patients each with about 5 scans a random split based on volume would have the model training on nearly identical data to what it is being validated and tested against. This would artificially increase the validation and test scores and overfit to the data given without much generalisability.

Using this split based on patient, there was about 70% in the training set, 15% in the validation and 15% in the test. This corresponded to 26 training patients, 6 test patients, and 6 validation patients. The idea was to try and use as many patients as possible to train the model. Use the validation patients as a way to save the best model generated (best minimum class Dice Similarity Coefficient). Test is completely separated and only used in predict.py. Further a fixed seed is used for both the 2D baseline and 3D model so that they are tested on the same splits and so that the randomisation removes all bias that can be generated from hand picking patients and volumes for each set.





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



