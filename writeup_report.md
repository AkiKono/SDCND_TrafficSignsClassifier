# **SDCND Project2: Traffic Sign Recognition**

## Classify German Traffic Signs by Training Deep Convolutional Neural Network

### DATA: [GTSRB dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)  

---

### **Project Summary**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Augment data
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* **Summarize the results with a written report**

Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

---

[//]: # (Image References)

[image1]: ./examples/RandomOriginalTrainingDataImages.png "Training Data Imanges"
[image2]: ./examples/BarChartGivenDataDistribution_Ratio.png "Distribution of Training Data"
[image3]: ./examples/LocalNormRGB.png "Local Normalization of RGB"
[image4]: ./examples/GRAY_RESCALE_RED_RESCALE.png "GRAY_RESCALE_RED_RESCALE"
[image5]: ./examples/SHADOW_COLOR.png "SHADOW_COLOR"
[image6]: ./examples/HIST.png "Histogram Equalization"
[image7]: ./examples/ErrorOriginalDataSet.png "ErrorOriginalDataSet"
[image8]: ./examples/OriginalDataSet.png "OriginalDataSet"
[image9]: ./examples/ErrorSplitDataSet.png "ErrorSplitDataSet"
[image10]: ./examples/SplitDataSet.png "SplitDataSet"
[image11]: ./examples/ErrorAugDataSet.png "ErrorAugDataSet"
[image12]: ./examples/AugDataSet.png "AugDataSet"
[image13]: ./examples/ErrorSplitAugDataSet.png "ErrorSplitAugDataSet"
[image14]: ./examples/SplitAugDataSet.png "SplitAugDataSet"
[image15]: ./examples/ErrorSameRatioDataSet.png "ErrorSameRatioDataSet"
[image16]: ./examples/SameRatioDataSet.png "SameRatioDataSet"
[image17]: ./examples/CLAHE.png "CLAHE"
[image18]: ./examples/CLAHE_.png "CLAHE_"
[image19]: ./examples/Rotate.png "Rotate"

[image99]: ./examples/Augmentation.png "Augmentation"



### Writeup Report

#### Provide a Writeup that includes all the [rubric points](https://review.udacity.com/#!/rubrics/481/view) and how you addressed each one.

1. Data Exploration
    * Dataset Summary
    * Exploratory Visualization
2. Design and Test a Model Architecture
    * Preprocessing
    * Model Architecture
    * Model Training
    * Solution Approach
3. Test a Model on New Images
    * Acquiring New Images
    * Performance on New Images
    * Model Certainty- Softmax Probabilities
4. Improvement on Deep Convolutional Neural Network

The final model achieved 97.4% accuracy in 12630 test dataset (324/12630 errors).
The training data accuracy was 98.9% and validation data accuracy was 98.7%.
The deep convolutional neural network architecture was designed based on LeNet architecture.

---

## 1. Data Set Summary & Exploration

### DATA SUMMARY:
**Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.**

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### EXPLORATORY VISUALIZATION:
**Include an exploratory visualization of the dataset and identify where the code is in your code file.**

The code for this step is contained in the third and forth code cells of the IPython notebook.  

Here is an exploratory visualization of the data set.
Random original training images are shown below.

![alt text][image1]

Variation of data in dataset:

Dataset contains some images that are too dark for human eyes to recognize the signs. Some images are burred. Some images have shades and/or reflections due to various lightning conditions. Shapes are warped and rotated. The scale of the signs in the images are also different.

The below is a bar chart showing how many image data each class contains, and their ratio.

![alt text][image2]

Some classes have over 2000 image data whereas most of others have less than 400 image data. Data argumentation scheme is used to ensure even distribution of dataset for training and validation. Details will be explained in the next section.

---

## 2. Design and Test a Model Architecture

### PREPROCESSING:
**Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.**

The code for this step is contained in the fifth code cell of the IPython notebook.

**i. EXPERIMENT**

In this section, 5 different preprocessing schemes are compared. The LeNet model architecture from LeNet.Lab is used with fixed hyperparameters for consistency in this experiment.

**Trial 1:**

Raw RGB image data values ranging from 0 to 255 were rescaled in the range from 0.1 to 0.9. The rescaling was done globally with the equation:  
0.1 + input*(0.9-0.1)/255 [equ.1]

The validation accuracy of the first trial was 92.9%.

**Trial 2:**

Raw RGB data was rescaled locally in the range 0.1-0.9 using the equation:   
0.1 + (input-min(inputs))x(0.9-0.1)/(max(inputs)-min(inputs))) [equ.2]

The validation accuracy of the second trial was 93.9%.

Local rescaling of the training data can increase validation accuracy of the model by 1% compared to trial 1. Since the total number of validation dataset is 4410, 1% improvement means 441 more successful classification, which is statistically significant. One possible reason is the whitening effect of local rescaling. The figure below compares original raw RGB images and locally rescaled images.

![alt text][image3]
Top: raw RGB image, Bottom: RGB image after local rescaling


**Trial 3:**

In addition to rescaled RGB images used in the trial 2, luminance Y grayscale images was added as the 4th channel in the dataset.

Luminance Y was calculated using the equation,

Y = 0.299*R + 0.587*G + 0.114*B  [equ.3]

Each of 3-channel RGB data is different from 1-channel grayscale data. Grayscale data differentiates color information from the original images and extract only the brightness from the RGB images.
Brightness or intensity scale captures the contours or shapes of objects. The hypothesis here is that the gray-scaling image data would help the model to identify the features such as edges and shapes in the images regardless of their colors, and such information is as important as color information.
To enhance the contrast of each grayscale image, rescaling was done locally with the equation 2. Figure below shows Y channel images captures the contours and shapes of objects better than R channel image in RGB.

![alt text][image4]
1st Row: original RGB image, 2nd Row: Y image, 3rd Row: locally rescaled Y image, 4th Row: locally rescaled R image in grayscale

The validation accuracy of the third trial was 95.2%.

Adding grayscaled image dataset as a 4th channel increased accuracy by 1.3%. It should be noted that depth of the first convolutional layer was increased by 1 due to the added 4th channel, which means there were more parameters in this trial than the previous trials.

The color channel information is valuable as they tend not be affected by shadows or different lighting conditions. The bottom image in the figure below is the R channel image which color values are least affected by the shadow compared to Y channel image in 3rd row where the intensity of the shadow is rather emphasized.

![alt text][image5]

1st Row: original RGB image, 2nd Row: Y image, 3rd Row: locally rescaled Y image, 4th Row: locally rescaled R image in grayscale

**Trial 4:**

Instead of rescaling using equation 2, the data values were normalized using mean values and standard deviation as in the equations shown below, all the rest are stayed the same as trial 3.

(input-mean(image))/std(image)  
image[image[:,:]> 3] =  3  
image[image[:,:]<-3] = -3  
(image/6+0.5)

The validation accuracy of the forth trial was 94.4%.

Normalization decreased the validation accuracy by 0.8% which is significant. One possible reason is that some information is lost when setting the values greater than 3 to 3 and setting the values less than -3 to -3. Also, normalization using standard deviation might reduced uniqueness of data distribution for different traffic signs.

**Trial 5:**  

In the trial 5, Histogram Equalization is used instead of normalization. The accuracy dropped to 83%.

![alt text][image6]
Top: original RGB image, Middle: Histogram Equalization on Y image, Bottom: Histogram Equalization on R image

**Trial 6:**

In the trial 6, Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied to Y image before rescaling. CLAHE applies intensity equalization on a local region of a image with limited contrast. It repeats this process until it covers the whole image. It greatly enhanced the visibility of signs especially when it is backlighted. In the figure below, images in odd rows are rescaled Y images with CLAHE, and images in even rows are the rescaled Y images without CLAHE.

![alt text][image17]

I have used 4x4 grid size with 2.0 clip limit. It took over 30 minutes to process 32x32 Y images with 4x4 grid size whereas other pre-processing took only about few minutes at most.

![alt text][image18]
1st Row: original RGB image, 2nd Row: R image, 3rd Row: G image, 4th Row: B image, 5th Row: Y image with CLAHE

The validation accuracy of the sixth trial was 95.9%.

**ii. CONCLUSION**

As a conclusion, the Trial 6 yielded the best result in this experiment. Locally rescaled RGB channels and locally rescaled luminance Y channel after CLAHE are the selected pre-processing method for this project.

Reference
["Should I normalize/standardize/rescale"](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)

**Raw RGB or Rescaled RGB**

I have tried with raw, unscaled RGB color channels and a rescaled Y channel data, and it resulted in less accuracy than the Trial 3, which used rescaled RGB and Y channels. The thought behind trying raw RGB color channels is that if RGB datasets are rescaled separately, unique combinations of RGB will be unbalanced so that the color information other than red, green, and blue get lost. It is because other colors are represented as a combination of RGB, e.g. Yellow=[255,255,0]. The possible explanation of rescaled RGB data performing better than raw RGB data could be that the most German traffic signs are strongly associated with red and blue colors. So for this specific traffic sign classifier project, emphasizing RGB color intensity gradient using per-channel rescaling led to better performance. Normalizing color channels however may not give good results in different classifying problems where colors like yellow or purple are key colors to identify objects.

Pierre Sermanet and Yann LeCun also suggested the use of normalized color channels in their paper, ["Traffic Sign Recognition with Multi-Scale Convolutional Networks"](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).


### PREPROCESSING:
**Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)**

The code for loading original dataset is contained in the first code cell of the IPython notebook. (The code for loading pre-processed dataset is contained in the "Load Pre-Processed Data" section before Model Architecture.)

The data was originally split into training, validation, and test datasets.

All the codes for pre-processing training and validation dataset can be found in "Pre-process the Data Set" section in Step2. In the first code cell in this section, the original training and validation datasets were merged together and shuffled. The combined dataset was split into training dataset and validation dataset in a specified ratio. In the next code cell, augmentation was applied.

Five different approaches for splitting and augmenting training and validation datasets were tested and compared, while the original testing dataset is kept unchanged and used to test final accuracy of a model in all trials for consistency.

**i. CONCLUSION**

Conclusion comes first. My final training dataset had 4000 images per 43 different classes summing up to 172000 total training images. My validation set had 750 images per 43 different classes summing up to 32250 total validation images. The ratio between Validation and Training dataset is 1:5. The size of test set is 12630 with different numbers of images in each class.

Figure below is the summary table of the 5 trials, and the final model.

|   |  Trial1|   Trial2|   Trial3|   Trial4|   Trial5|  Final |
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:|
|Test Accuracy      |91.5%|90.7%|89.8%|92.5%|92.7%| 97.4%| 	
|Validation Accuracy|92.4%|94.1%|92.1%|94.7%|98.9%| 98.7%|  	
|Training Accuracy  |99.6%|99.7%|98.9%|99.7%|99.4%| 98.9%|  	
|# of data in Test|12630|12630|12630|12630|12630|12630|   	
|# of data in Validation|4410|11871|4410|11871|4300|32250|   	
|# of data in Training|34799|27338|56038|50755|43000|172000|   	
|Ratio of Validation:Training|1:8|1:2.3|1:13|1:4.3|1:10|1:5|   	
|Augmentation|||O|O|O|O|   	
|# of data in Validation increased||O||O||O|
|Same # of Data in Each Class|||||O|O||

Trial1  
![alt text][image8]
Trial2  
![alt text][image10]
Trial3  
![alt text][image12]
Trial4  
![alt text][image14]
Trial5  
![alt text][image16]

**ii. EXPERIMENT**

EXPRIMENT ANALYSIS:  
Training accuracies were high enough for all trials, so underfitting is not a concern. Rather, overfitting is the problem. Comparing Trial 1 through 4, validation accuracy increased from 92% to 94% when the total number of validation dataset was increased. Both Trial 2 and Trial 4 had exactly the same number of validation data. The difference between Trial 2 and 4 is the total number of Training dataset by augmentation. Trial 4 had almost twice as much training data as Trial 2 had. However, the difference in validation accuracy was only 0.6%. The ratio of validation and training dataset is not so much correlated to overfitting, but different number of available validation data for different classes is strongly correlated to overfitting. In other words, the model is overfitted to some classes that have many validation data and underfitted to some classes that have small validation data. So in Trial 5, all classes had the same number of validation data and training data. Then, the accuracy of validation dataset increased to 98.9%. The difference in accuracy between validation and training was decreased from 2.2% in Trial 4 to 0.5% in Trial 5. Moreover, Trial 5 used less number of validation and training data but marked higher test accuracy than Trial 4 did.

Still, test accuracy is low compared to validation and training accuracy, so the model is overfitted to training and validation data. This issue is readdressed in the discussion of model architecture.

AUGMENTATION ISSUE:  
Going back to the summary table, comparing Trail 1 and Trial 3, the difference is the increased number of training data by augmentation; however, neither validation nor test accuracies were improved which indicates augmentation is just adding sampling noise to the model and needs more improvements. This will be the next topic.

AUGMENTATION:  
The number of training and validation dataset was increased by augmentation in two processes. One is to increase traffic sign images with as small noise as possible. The second augmentation is intentionally adding more noise to original images to encourage generalization.

The code for the first augmentation process can be found in the 8th code cells or in the section "Pre-process the Data Set: First Augmentation by Flipping, Rotation, Shearing, and Translation." Axisymmetric and/or radial symmetric traffic signs were simply flipped and/or rotated. For example, rotary sign is radially symmetric so that it was rotated in 120 and 240 degrees to triple the number of images.

![alt text][image19]

This augmentation technique was inspired by [ALEX SARAVOITAU](http://navoshta.com/traffic-signs-classification/).

Slight shear, rotation, and translation were added to increase the size of base images for all classes to be more than 1700 images.  

These 1700 images are then split.
The code for splitting dataset can be found in the 10th cell, or in the "Pre-process the Data Set: Split dataset into training and validation dataset."

Then, 450 images per class were randomly chosen and assigned as validation dataset in the splitting process. At most 1250 images per class were randomly chosen and assigned as training dataset.

The second augmentation is applied to these seed dataset and increased to 750 images per class for validation and 4000 images per class for training.

The code for second augmentation can be found in the section "Pre-process the Data Set: Second Augmentation by Sharpening, Whitening, Translation, Rotation, Shearing, and Gaussian Blur."

Image editing methods used here were sharpening, whitening, translation, rotation, shearing, zooming out, and Gaussian blur. These were randomly applied to images with random intensities in a specified range.  

Here are the random images after augmentation.

![alt text][image99]


CROSS VALIDATION:  
Cross validation is another technique which often used in the case where only few hundreds data are available. It is highly possible that cross validation can improve the model and accuracy, but it was not applied this time.


### MODEL ARCHITECTURE:
**Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.**

The code for my final model is located in the "Model Architecture" section of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| Preprocessed 32x32x4 RGBY image  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x36 	|
| ELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 8x8x54 	|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x54 		  		    |
| Flattened     		|           									|
| Dropout       		| Pass Probability: 0.9							|
| Fully connected		| Input: 864, Output: 480    					|
| Fully connected		| Input: 480, Output: 168    					|
| Fully connected		| Input: 168, Output: 43    					|
| Softmax				|           									||

Figure below is the visualization of the graph from TensorBoard.

 ![alt text][image91]

###  MODEL TRAINING:
**Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.**

The code for training the model is located in the "Train, Validate, and Test the Model" section in the ipython notebook.

TRAININT STEPS:  
 * Deep feedforward neural network calculates probabilities as logits.
 * Softmax is then applied to logits and compared with One-Hot encoding to measure how well the model is classifying dataset.
 * Cross-entropy is used as a cost function to measure the differences between softmax and One-Hot encoding. The derivative of Cross-entropy with respect to weights gives a simple multiplication of the error and the inputs to the neuron. This ensures the fast convergence of learning.

**Batch size**  
Mini-batch stochastic gradient descent was used to update weights and biases in back-propagation. Batch size used for the final model results was 32.

**Number of Epochs**  
The number of Epochs used for the final model results was 5. The early stoping prevented over fitting.

**Sigma**  
Weights initialization variance was set to 0.1.

**Optimizer**  
Adaptive Moment Estimation or Adam optimizer was used to optimize gradient descent process. An optimum learning rate and an optimum added momentum are calculated for each parameter and updated in every batch run. This adaptive optimal step-size is calculated from the ratio between averaged past gradients and square root of squared averaged past gradients, such that the optimum step-size is independent of the magnitude of gradients. I used recommended starting learning rate 0.001 and other default parameters. (beta1=0.9, beta2=0.999, epsilon=1e-08) I have tried epsilon = 1e-04 but it didn't improved learning performance.

Figure below is the plot of accuracy and cross-entropy loss function with respect to progress in training.

 ![alt text][image90]


References:  
["An overview of gradient descent optimization algorithms"](http://sebastianruder.com/optimizing-gradient-descent/index.html#fn:18)  
["ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"](https://arxiv.org/pdf/1412.6980.pdf)   
["Improving the way neural networks learn"](
http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function)  

### SOLUTION APPROACH:
**Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.**

[//]: # (Image References)
[image20]: ./examples/ELU.png "ELU"
[image21]: ./Results/DyingReLU.png "DyingReLU"

[image50]: ./traffic-signs-from-web/0.png "TrafficSignsFromWeb1"
[image51]: ./traffic-signs-from-web/1.png "TrafficSignsFromWeb2"
[image52]: ./traffic-signs-from-web/2.png "TrafficSignsFromWeb3"
[image53]: ./traffic-signs-from-web/3.png "TrafficSignsFromWeb4"
[image54]: ./traffic-signs-from-web/5.png "TrafficSignsFromWeb5"
[image55]: ./traffic-signs-from-web/7.png "TrafficSignsFromWeb6"
[image56]: ./traffic-signs-from-web/11.png "TrafficSignsFromWeb7"
[image57]: ./traffic-signs-from-web/12.png "TrafficSignsFromWeb8"
[image58]: ./traffic-signs-from-web/13.png "TrafficSignsFromWeb9"
[image59]: ./traffic-signs-from-web/14.png "TrafficSignsFromWeb10"
[image60]: ./traffic-signs-from-web/18.png "TrafficSignsFromWeb11"
[image61]: ./traffic-signs-from-web/23.png "TrafficSignsFromWeb12"
[image62]: ./traffic-signs-from-web/24.png "TrafficSignsFromWeb13"
[image63]: ./traffic-signs-from-web/25.png "TrafficSignsFromWeb14"
[image64]: ./traffic-signs-from-web/29.png "TrafficSignsFromWeb15"
[image65]: ./traffic-signs-from-web/31.png "TrafficSignsFromWeb16"
[image66]: ./traffic-signs-from-web/33.png "TrafficSignsFromWeb17"
[image67]: ./traffic-signs-from-web/35.png "TrafficSignsFromWeb18"
[image68]: ./traffic-signs-from-web/40.png "TrafficSignsFromWeb19"

[image70]: ./Results/ErrorTest_97_4.png "Results1"
[image71]: ./Results/ErrorTestImage_97_4.png "Results2"
[image73]: ./Results/predict1.png "Results4"
[image74]: ./Results/predict2.png "Results5"
[image75]: ./Results/predict3.png "Results6"
[image76]: ./Results/predict4.png "Results7"
[image77]: ./Results/predict5.png "Results8"
[image78]: ./Results/FeatureMap.png "Results9"
[image79]: ./Results/ErrorLabels.png "Results10"

[image80]: ./Results/sigma05_1.png "Results11"
[image81]: ./Results/sigma05_2.png "Results12"
[image82]: ./Results/sigma05_3.png "Results13"
[image83]: ./Results/sigma05_4.png "Results14"
[image84]: ./Results/sigma05_5.png "Results15"
[image85]: ./Results/sigma10_1.png "Results16"
[image86]: ./Results/sigma10_2.png "Results17"
[image87]: ./Results/sigma10_3.png "Results18"
[image88]: ./Results/sigma10_4.png "Results19"
[image89]: ./Results/sigma10_5.png "Results20"

[image90]: ./Results/accuracy_loss.png "Results21"
[image91]: ./Results/model.png "Results22"

The code for calculating the accuracy of the model is located in the second cell of "Train, Validate and Test the Model" section.

My final model results were:
* training set accuracy of 98.9%
* validation set accuracy of 98.7%
* test set accuracy of 97.4%

* **What was the first architecture that was tried and why was it chosen?**

 The first architecture tried was LeNet from LeNet.Lab. The first layer consists of 5x5 convolution with 1 stride and VALID padding, followed by ReLU activation and 2x2 max pooling. The structure of second layer is the same as the first layer. The output from second layer was flattened and fed to the first of three fully connected layers which reduced the number of output from 900 to 43, which is then evaluated with one hot encoding after softmax. Total number of parameters are (5x5x4x18+18) + (5x5x18x36+36) + (900x480) + (480x168) + (168x43) = 1818 + 16236 + 17550 + 432000 + 80640 + 7224 = 555468.

 With the pre-processing and augmentation, the model reached 99.9% accuracy on training, 99.8% on validation, and 93.3% on testing. Overfitting to the training and validation dataset was the problem and the model needs to be generalized.

* **What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem?**

 A convolution layer works well with image classifier because convolution is designed to find patterns in locally depended data. 32x32x3 Image data are meaningful only when they are properly positioned in a 32x32 image frame. Filters in convolutional layers extract patterns, such as edges or lines, in the image data locally. This is why convolution layer is suitable to image processing problems.  

* **How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.**

 The model had many errors on classifying speed limit traffic signs. To retain details of the original images, max pooling on the second layer was removed and 3x3 convolutional layers with stride of 1 and 54 output depth is added. Then maxpooling is used to reduce the width of data and fully connected layers further reduced it down to 43. Total number of parameters are (5x5x4x18+18) + (5x5x18x36+36) + (3x3x36x54+54) + (864x480) + (480x168) + (168x43) = 1818 + 16236 + 17550 + 432000 + 80640 + 7224 = 538188. Total number of parameters is decreased by 17280. The model accuracy increased 1%.

 Adding dropout after fully connected layers increased accuracy on testing data by 0.8%. Dropout deactivates neurons in hidden layers to let the model adopt to generalized training. 0.1 probability was used.

 Dying ReLU was a problem as you can see in the figure below.
 ![alt text][image21]

 All activation function was changed from Rectified Linear Units (ReLU) to Exponential Linear Units (ELU). The plot of ELU function can be found in the figure below.

 ![alt text][image20]

 The idea of having negative activation is to make the mean values after activations closer to zero for all activations throughout the network so to prevent internal covariate shift, the same effect as batch normalization. What is notable here is that ELU outperformed ReLU trained with batch normalization, according to the author of ["FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)."](https://arxiv.org/pdf/1511.07289.pdf)
 Just by changing from ReLU to ELU, the learning converged faster and the final test accuracy increased 0.6%. Also, not having zero activations, dying ReLU problem is not a concern for ELU. Further research on Parametric ELU can be useful as it addresses vanishing gradients in the negative arguments. Currently Tensorflow does not have Parametric ELU function. Further discussions on weights initialization for ELU can be found in the paper. This is essential when the model is deeper (> 8 conv layers). When I tried 6 convolutional layers, the accuracy of the model decreased from 98% to 23% in one Epoch. It seems to be diverged.  

* **Which hyperparameters were tuned? How were they adjusted and why?**

    * **Batch Size:**   
    For mini-batch SGD, parameters such as weights and biases are updated using the averaged gradients of one mini-batch. Mini-batch sized are commonly between 32-512. I have tried 64 and 256. The accuracies of training, validation, and testing were all higher when batch size 64 was used. The effect of batch sizes on network using ADAM optimizer is explored in the article ["ON LARGE-BATCH TRAINING FOR DEEP LEARNING: GENERALIZATION GAP AND SHARP MINIMA"](https://arxiv.org/pdf/1609.04836.pdf). The author concluded that large batch method causes overfitting and tends to attracted to closest local minima near initial point, which is consistent with my results.

    * **Number of Epochs:**   
    The training was stopped when the validation accuracy starts to decrease. This is the sign of overfitting.

    * **Weight Initialization Variance Sigma:**  
    Two different variances sigma = 0.05 and sigma = 0.1 for weights initialization were compared. Figures below shows the changes in histogram of weights while training.

    **Sigma=0.05 Histogram of Weights and Biases**

    ![alt text][image80]
    ![alt text][image81]
    ![alt text][image82]
    ![alt text][image83]
    ![alt text][image84]

    **Sigma=0.10 Histogram of Weights and Biases**

    ![alt text][image85]
    ![alt text][image86]
    ![alt text][image87]
    ![alt text][image88]
    ![alt text][image89]

    When sigma=0.05, the variance of the weights are increasing, whereas when sigma=0.1, the changes in variance is less significant. We want the variance to remain the same. So the sigma=0.1 is chosen for the final model.

* **What architecture was chosen? Why did you believe it would be relevant to the traffic sign application?**

 I have tried Inception Module as I thought different filter sizes applied to the first input images can detect both large features, such as circler or trianguler shape of traffic sign boards, and small features, such as numbers or illustrations on the center of the boards. However, it was computationally heavy and the performance was poor compared to simple LeNet structure. Increasing epochs from 10 to 30 did not improve accuracy. More parameter turning and design exploration is required to make a use of Inception Module. So I decided to stick with LeNet architecture as it seems a better starting point for deep learning novice.

 I have tried using only convolutional layers except the last fully connected layer reducing outputs to 43. This was inspired by the article ["Striving for Simplicity: The All Convolutional Net"](https://arxiv.org/abs/1412.6806). Although max pooling greatly reduce the number of neurons, (For example, 75% reduction with 2x2 max pooling with stride of 2), it also loses a lot of information which might be important. So the max pooling layer is replaced by convolutional layers with stride of 2 or 3. Also, fully connected layers were replaced by 1x1 convolution to reduce the depth. Again, same as Inception Module, I couldn't train deep convolutional neural network (8 convolutional layers) well enough so that the accuracy did not improve compared to LeNet model with max pooling and fully connected layers.

 The article
["Deep Convolutional Neural Network Design Patterns"](https://openreview.net/pdf?id=SJQNqLFgl) is now on my reading list for further research.

---

## 3. Test a Model on New Images

### ACQUIRING NEW IMAGES:
**Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.**

Here are German traffic signs that I found on the web:

![alt text][image54] ![alt text][image52]
![alt text][image55] ![alt text][image66] ![alt text][image56] ![alt text][image62] ![alt text][image65]
![alt text][image59] ![alt text][image58]
![alt text][image57] ![alt text][image60] ![alt text][image68]
![alt text][image67] ![alt text][image50] ![alt text][image64]
![alt text][image51] ![alt text][image53] ![alt text][image63]
![alt text][image61]

First 10 images are from the web. The rest of the images are from the GTSRB dataset which Pierre Sermanet and Yann LeCun presented that their model predicted wrong. I was curios to know how much confidence my model has on the those GTSRB images that are difficult to classify.

Variations in images from the web:

Coarse resolution.

![alt text][image54] ![alt text][image52]
![alt text][image55]

Severely warped.   

![alt text][image66] ![alt text][image56]

Occulusion due to objects in front.  

![alt text][image62] ![alt text][image65]
![alt text][image59]

Deformation.  

![alt text][image58]

Severe reflections of lights.  

![alt text][image57]

All these variations can make classification difficult.



### PERFORMANCE ON NEW IMAGES:
**Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).**

The code for making predictions on my final model is located in the second cell of the Ipython notebook in the "Step 3: Test a Model on New Images" section.

The model was able to correctly guess 13 of the 19 traffic signs, which gives an accuracy of 68% which is lower than the accuracy of test dataset. More detailes discussions on this result can be found in the next section.

### MODEL CERTAINTY - SOFTMAX PROBABILITIES:
**Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)**

The code for making predictions on my final model can be found in the section "Output Top 5 Softmax Probabilities For Each Image Found on the Web" in the Step 3.

Right picture shows the actual traffic sign image with the correct label shown on the top of image. The middle 5 pictures are predicted top five traffic signs. The right bar graph shows how confident the model is for predicted top fives.  

![alt text][image73]

It made a mistake on "Speed limit 60km/h," but over all it is making a good predictions based on colors and shapes.

![alt text][image74]

The lowest confidence the model had was 5% on the "Right-of-way at the next intersection" image.

![alt text][image75]

The model made a wrong prediction on "Speed limit 50km/h" as "Speed limin 100km/h" with 100% confidence.

![alt text][image76]

These are from GTSRB datasets except "Road narrows on the right." Not so surprising that the predictions are correct and the model is 100% confident.

![alt text][image77]

Here again, the model made a wrong prediction on "Speed limit 80km/h" as "Speed limit 100km/h" with high confidence, and it made a wrong prediction on "Speed limit 100km/h" as "No Vehicles" with high confidence.

In other words, the model has low recall AND low precision on "Speed limit 100km/h."

The below is the feature map for "Speed limit 100km/h" after the first convolutional layer with 5x5 filter and 1x1 stride was applied. As expected, the number "100" is not clear in the feature map, which means the model is not picking up numbers. Numbers inside the red circle are the only key feature to classify different Speed Limit traffic signs.

![alt text][image78]

Figure below shows the number of prediction errors in each label. The labels on Speed Limit traffic signs are from 0 to 8 for 20km/h limit to 120km/h limit. The errors on those speed limit signs are relatively high.

![alt text][image70]

Below are the images the model failed to predict.

![alt text][image71]

Some of the images have severe occulusion, high contrast lighting, blur, warp, and/or too small in scale. Some, however, with no apparent obstackles for classification.

The final accuracy of the model was 97.4%. This is not a satisfactory result for a practical application. The model needs more improvement.

---

## 4. Improvement on Deep Convolutional Neural Network

Below is a list of topics I would like to learn more to further improve the model.

* ["Deep Convolutional Neural Network Design Patterns"](https://openreview.net/pdf?id=SJQNqLFgl)
* Instance base learning for classifying
* [Spatial Transformer 99.61% accuracy](http://torch.ch/blog/2015/09/07/spatial_transformers.html)
* [DNN 99.46% accuracy](http://people.idsia.ch/~juergen/nn2012traffic.pdf)
* Papers that are classifying GTSRB dataset with high accuracy often uses the technique called Histograms of Oriented Gradients (HOG).
And it seems we are going to learn it in vehicle detection, so I will not try to implement it in this project.
* locality-constrained linear coding (LLC)
* linear support vector machine (SVM)
* Gabor filter feature and local binary pattern (LBP)
* Scale-invariant feature transform (SIFT)
* Spatial Pyramid Matching (SPM)
* ["Traffic sign classification using two-layer image representation"](http://ieeexplore.ieee.org/abstract/document/6738774/)
* ["Traffic Sign Recognition Using Complementary Features"](http://ieeexplore.ieee.org/abstract/document/6778312/)
* Random Forest Decision Tree technique
Histogram of Gradients
["Reference"](https://link.springer.com/chapter/10.1007/978-3-319-48896-7_20)
* Implementation of Local Receptive Field
"Fully connected layers are often over-parameterized for vision tasks, and a more appropriate middle ground would be to use a "local receptive field", which (informally) maintains separate parameters for each input (as in a fully connected layer), but only combines values from "nearby" inputs to produce an output (as in a convolution). Unfortunately, TensorFlow doesn't yet contain an implementation of local receptive fields, but adding support for them would be a useful project."
[Reference](http://robotics.stanford.edu/~ang/papers/nips11-SelectingReceptiveFields.pdf)
* ["Uncertaintity and Dropout"](https://www.google.com/search?rlz=1C5CHFA_enUS732US733&espv=2&q=weights+variance+changes+as+the+model+is+trained+&oq=weights+variance+changes+as+the+model+is+trained+&gs_l=serp.3...25247.37177.0.37529.51.51.0.0.0.0.144.4785.39j10.49.0....0...1c.1.64.serp..2.34.3369...0j35i39k1j0i67k1j0i131k1j0i20k1j0i22i30k1j0i22i10i30k1j33i21k1j33i160k1.TYhETYHCFeA)
