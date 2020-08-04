# Image Colorization 
Chonnam National University, Gwangju, South Korea<br/>
Author: **Tran Nguyen Quynh Tram**<br/>
Papers: <br/>
1. Colorization of Natural Scene Image by Using a U-net <a href="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/papers/tnqtram_kism20_Colorization_of_Natural_Scene_Image_by_Using_a_U-net.pdf">Paper</a> <a href="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/papers/tnqtram_kism20_Colorization_of_Natural_Scene_Image_by_Using_a_U-net_slides.pdf">Slide</a>


2. MLEU: Multi-Level Embedding U-Net for Fully Automatic Image Colorization <a href="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/papers/tnqtram_icmlsc20_Multi_Level_Embedding_Image_Colorization.pdf">Paper</a> <a href="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/papers/tnqtram_icmlsc20_Multi_Level_Embedding_Image_Colorization_slides.pdf">Slide</a>
<br/>(https://dl.acm.org/doi/abs/10.1145/3380688.3380720)

## Problems
1. Accuracy is not convergence
2. Red noise in colorized image
3. Report: <a href="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/problem01/tnqtram_problems_image_colorization.pdf">Links</a>

## Training History
1. Using RMSProp, Adam with Step Decay, or Constant or Cycle Learning Rate
2. Training and validating on Coco-Stuff

**(1a) Validation Ground-truth**<br/>
<image src="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/problem01/images/pred_preview_test_soft_images_0055_18.43.jpg" height=300/>

**(1b) Validation Prediction for Soft-Encoding**<br/>
<image src="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/problem01/images/pred_preview_valid_reg_images_0055_22.12.jpg" height=300/><br/>
<font color="Red">Results (with Red Noise and some regions are not colorized)</font>

**(1c) Validation Prediction for Regression**<br/>
<image src="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/problem01/images/pred_preview_valid_soft_images_0055_18.83.jpg" height=300/>

** Training History: Loss **<br/>
<image src="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/problem01/images/logs_loss.jpg" height=300/>

** Training History: Accuracy **<br/>
<image src="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/problem01/images/logs_acc.jpg" height=300/>
<font color="Red">Accuracy is not convergence at Column 1</font>

** Predictions **<br/>
<image src="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/problem01/images/predictions.png" height=300/>
<font color="Red">Results (with Red Noise and some regions are not colorized)</font>

** Results **<br/>
<image src="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/problem01/images/results.png" height=300/>
<font color="Red">Results (with Red Noise and some regions are not colorized)</font>

## Model
<image src="https://raw.githubusercontent.com/tramtran2/prlab_image_colorization/master/problem01/images/models.png" height=300/>

## Source Codes:
** Loss Function:** https://github.com/tramtran2/prlab_image_colorization/blob/master/sources/image_colorization/mleu_train/losses.py<br/>
** Soft-Encoding Function:** https://github.com/tramtran2/prlab_image_colorization/blob/master/sources/image_colorization/datasets/quantized_colors/colorized_soft_encoding.py<br/>
** Decoding Function:** https://github.com/tramtran2/prlab_image_colorization/blob/master/sources/image_colorization/datasets/quantized_colors/decode_v1.py<br/>
** Models: ** https://github.com/tramtran2/prlab_image_colorization/blob/master/sources/image_colorization/mleu_train/models/zhang_models.py<br/>
