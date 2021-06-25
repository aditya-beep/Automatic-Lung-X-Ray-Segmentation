# Automatic-Lung-X-Ray-Segmentation
## Introduction
Segmentation of Chest X-Rays (CXRs) plays a crucial role in computer-aided diagnosis of chest X-Ray (CXR) images. CXRs are one of the most commonly prescribed medical imaging procedures with the voluminous CXR scans placing significant load on radiologists and medical practitioners. Automated organ segmentation is a key step towards computer-aided detection and diagnosis of diseases from CXRs. In this paper, we propose a deep convolutional network model trained in an end-to-end setting to facilitate fully automatic lung segmentation from anteroposterior (AP) or posteroanterior (PA) view chest X-Rays, which is based on the UNet++ model architecture, using Efficient Net B4 as the encoder and residual blocks from ResNet to create our decoder blocks. Our network learns to predict binary masks for a given CXR, by learning to discriminate regions of organ, in this case the lungs, from regions of no organ and achieves very realistic and accurate segmentation. In order to generalize our model to CXRs we also employ image pre-processing techniques like top-hat bottom-hat transform, Contrast Limited Adaptive Histogram Equalization (CLAHE), in addition to a couple of randomized augmentation techniques. The datasets chosen 
by us are also critical components in various computer-aided diagnosis of CXR  algorithms.

## Methodology

**IMAGE AUGMENTATION:**  Since a limited dataset is an obstacle in training deep convolutional neural networks, we can rectify this problem using image augmentation which is nothing but simply creating new images by making small modifications in the previous data set, to provide more images for better training and testing
Some of the modifications are:
 - Rotation – Rotating the image with a very small amount of angle probably 0.5 degree will create a new image. We just have to make sure that while doing this rotation the boundaries of lungs and edges do not go out of the image boundary
 - Width shift- images are randomly shifted on the horizontal axis by a fraction of total width
 - Height shift - Images are randomly shifted on the vertical axis by a fraction of the total height
 - Shearing – It is basically slanting the image, it is different from rotation .In this we fix one axis , and then stretch the image at a certain angle .
 = Horizontal flip – In horizontal flipping, images are flipped randomly , by doing this modification , the model will be able to better segment chest radiographs of front as well as back.

**OPTIMIZATION ALGORITHM:**  An optimization algorithm is arguably one of the most important tools of deep learning. These algorithms are responsible for training the deep convolutional neural network (DCNN) by updating the parameters of the network so that it learns to minimize an objective function aka the loss function. In our proposed approach, we employ the Nadam optimization algorithm. The Nadam optimizer has been shown to perform well for medical segmentation tasks and leads to a faster convergence to the local minima of the chosen loss function.

**LOSS FUNCTION:**  Deep-learning segmentation frameworks rely not only on the design of the network architecture but also on the type and complexity of the driving loss function. When we train our DCNN, we are essentially trying to run an optimization algorithm (in our case, Nadam) to minimize the chosen loss function. We realized the need of a specialized loss function which appropriately assigns weights to each class in order to balance the bias so that the model can better segment the lungs by providing additional emphasis on learning to classify all lung-related voxels which are a minority.
Our proposed novel loss function termed ‘Penalty Combo Loss’ (PCL) is defined as: LPCL = αLBCE + βLPGDL. The proposed PCL loss is essentially a weighted sum of Binary Cross-Entropy loss and Penalty Generalized Dice Loss where α and β are the weights assigned to the binary cross-entropy loss and penalty generalized dice loss functions respectively. α and β are hyperparameters which require fine-tuning depending on the application domain. The fine-tuning can be trivially accomplished using a holdout/validation set.
 - Binary cross-entropy loss (LBCE) is defined as: **𝑳𝑩𝑪𝑬= − 𝟏𝑵Σ𝒈𝒊𝒍𝒐𝒈(𝒑𝒊) + (𝟏−𝒈𝒊) 𝒍𝒐𝒈(𝟏−𝒑𝒊) 𝑵𝒊=𝟏**
 - Penalty Generalized Dice Loss (PGDL) is defined as: **𝑳𝑷𝑮𝑫𝑳= 𝑳𝑮𝑫𝑳/(𝟏+𝒌(𝟏−𝑳𝑮𝑫𝑳))**

**LEARNING RATE SCHEDULE:** The learning rate is considered to be one of the most important hyperparameters that directly affects the optimization process of a deep neural network (DNN) alongside model training and generalization. It is a positive scale factor which basically supervises the celerity of network convergence in order to reach the global minima by navigating through the non-convex loss function’s high-dimensional spatial surface. This convergence is often affected/delayed due to entrapment of the optimizer function at multiple aberrations such as local minima, saddle points etc.

## Result 
![image](https://user-images.githubusercontent.com/62646784/123479483-a1fce300-d61e-11eb-9c1a-9287f778ce2c.png)            ![image](https://user-images.githubusercontent.com/62646784/123479540-b50fb300-d61e-11eb-83b7-67045c0ebfc9.png)                                                                                                ![image](https://user-images.githubusercontent.com/62646784/123479640-dc668000-d61e-11eb-9f1b-4285d39a7261.png)


## CONCLUSION AND FUTUREWORK

We demonstrate in our lung segmentation method that the problem of dense abnormalities in Chest X-rays can be efficiently addressed by performing a reconstruction step based on a deep convolutional neural network model.

In the future, we plan on extending our project’s usage for aiding doctors in COVID diagnosis and treatment. We also plan on working on making a web-app for the project, where doctors/technicians can simply upload the Chest X rays of patients, and our Neural Network will predict the segmented lung region and return it to the user. This will surely help in making it much more accessible to doctors anywhere in this country, as only an internet connection would be required to diagnose anyone anywhere in India.
