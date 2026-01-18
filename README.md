<h2>TensorFlow-FlexUNet-Image-Segmentation-Skin-Cancer  (2026/01/19)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Skin Cancer</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and  <a href="https://www.kaggle.com/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation">Skin Cancer: Lesions Segmentation</a> dataset.
<br><br>
<hr>
<b>Actual Image Segmentation for Skin-Cancer Images of 600x450 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/images/ISIC_0024340.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/masks/ISIC_0024340.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test_output/ISIC_0024340.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/images/ISIC_0024406.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/masks/ISIC_0024406.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test_output/ISIC_0024406.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/images/ISIC_0024673.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/masks/ISIC_0024673.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test_output/ISIC_0024673.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://www.kaggle.com/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation">Skin Cancer: Lesions Segmentation</a><br>
HAM10000 multi-source dermatoscopic images of pigmented skin lesions.
<br><br>
<b>About Dataset</b><br>
Training of neural networks for automated diagnosis of pigmented skin lesions is hampered by the small size and lack of diversity of 
available datasets of dermatoscopic images. <br>
We tackle this problem by releasing the HAM10000 ("Human Against Machine with 10000 training images") dataset.<br>
 We collected dermatoscopic images from different populations, acquired and stored by different modalities.
<br>
The final dataset consists of 10015 dermatoscopic images which can serve as a training set for academic machine learning purposes.
<br>
Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions:
<br>
<ul>
<li><b>AKIEC</b> - actinic keratoses and intraepithelial carcinoma / Bowen's disease</li>
<li><b>BCC</b> - basal cell carcinoma</li>
<li><b>BKL</b> - benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus-like keratoses)</li>
<li><b>DF</b> - dermatofibroma</li>
<li><b>MEL</b> - melanoma</li>
<li><b>NV</b> - melanocytic nevi</li>
<li><b>VC</b> - vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and haemorrhage)</li>
</ul>
<br>
More than 50% of lesions are confirmed through histopathology, the ground truth for the rest of the cases is either follow-up examination (follow_up), 
expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). <br>
The dataset includes lesions with multiple images, which can be tracked by the lesion_id-column within the metadata file.
<br><br>
<b>License</b><br>
<a href="https://spdx.org/licenses/CC-BY-NC-4.0.html">Creative Commons Attribution-NonCommercial 4.0 International License</a>
<br>
<br>
<h3>
2 Skin-Cancer ImageMask Dataset
</h3>
 If you would like to train this Skin-Cancer Segmentation model by yourself,
please down load master dataset 
<a href="https://www.kaggle.com/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation">Skin Cancer: Lesions Segmentation</a>.
<br><
We used a Python script <a href="./generator/split_master.py">split_master.py </a>to spit the master dataset into test, train and valid subset.<br>
<pre>
./dataset
└─Skin-Cancer
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Skin-Cancer Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Skin-Cancer/Skin-Cancer_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Skin-Cancer TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Skin-Cancer/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Skin-Cancer and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 2
base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8

dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Skin-Cancer 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Skin-Cancer 1+1
rgb_map = {(0,0,0):0,  (255, 255, 255):1, }       
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (11,12,13)</b><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (24,25,26)</b><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stoppd at epoch 26 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/train_console_output_at_epoch26.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Skin-Cancer/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Skin-Cancer/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Skin-Cancer</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Skin-Cancer.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/evaluate_console_output_at_epoch26.png" width="880" height="auto">
<br><br>Image-Segmentation-Skin-Cancer

<a href="./projects/TensorFlowFlexUNet/Skin-Cancer/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Skin-Cancer/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.098
dice_coef_multiclass,0.9451
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Skin-Cancer</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Skin-Cancer.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Skin-Cancer/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Skin-Cancer Images of  600x450 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/images/ISIC_0024357.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/masks/ISIC_0024357.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test_output/ISIC_0024357.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/images/ISIC_0024406.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/masks/ISIC_0024406.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test_output/ISIC_0024406.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/images/ISIC_0024441.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/masks/ISIC_0024441.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test_output/ISIC_0024441.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/images/ISIC_0024482.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/masks/ISIC_0024482.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test_output/ISIC_0024482.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/images/ISIC_0024530.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/masks/ISIC_0024530.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test_output/ISIC_0024530.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/images/ISIC_0024673.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test/masks/ISIC_0024673.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Skin-Cancer/mini_test_output/ISIC_0024673.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. ISIC 2017 - Skin Lesion Analysis Towards Melanoma Detection</b><br>
Matt Berseth<br>
<a href="https://arxiv.org/ftp/arxiv/papers/1703/1703.00523.pdf">
https://arxiv.org/ftp/arxiv/papers/1703/1703.00523.pdf
</a>
<br><br>
<b>2. ISIC Challenge Datasets 2017</b><br>
<a href="https://challenge.isic-archive.com/data/">
https://challenge.isic-archive.com/data/
</a>
<br><br>
<b>3. Skin Lesion Segmentation Using Deep Learning with Auxiliary Task</b><br>
Lina LiuORCID,Ying Y. Tsui andMrinal MandalM<br>
<a href="https://www.mdpi.com/2313-433X/7/4/67">
https://www.mdpi.com/2313-433X/7/4/67
</a>
<br><br>
<b>4. Skin Lesion Segmentation from Dermoscopic Images Using Convolutional Neural Network</b><br>
Kashan Zafar, Syed Omer Gilani, Asim Waris, Ali Ahmed, Mohsin Jamil,<br>
Muhammad Nasir Khan and Amer Sohail Kashif<br>
<a href="https://www.mdpi.com/1424-8220/20/6/1601">
https://www.mdpi.com/1424-8220/20/6/1601
</a>
<br><br>
<b>5.  Image-Segmentation-Skin-Lesion</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/Image-Segmentation-Skin-Lesion">
https://github.com/sarah-antillia/Image-Segmentation-Skin-Lesion</a>
<br><br>
<b>6. Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer">
Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer
</a>
<br>
<br>
<b>7. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
