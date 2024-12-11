# Skin Tone Detection for Cosmetic Product Recommendation
In the ever-changing cosmetics industry, consumers often struggle to choose products that match their skin tone, particularly for items such as foundation. Our project aims to develop a machine learning framework that addresses this challenge by automatically analyzing skin tone from face images, assisting consumers in selecting cosmetic products tailored to their specific skin tone.

Our approach consists of two main stages:
1. **Skin Pixel Extraction:** We will implement a convolutional neural network (CNN) to classify individual pixels in an image as skin or non-skin, building upon the methodology outlined in [1]. For this, we will utilize the ECU skin dataset [2], which provides diverse skin images for training and evaluation. As baseline comparisons, we will also implement traditional approaches such as YCbCr, HSV and HSCbCr color space methods. We plan to enhance the existing pipeline by:
    - Experimenting with alternative neural network architectures, including established models like LeNet5 and ResNet, to improve classification accuracy.
    - Conducting hyperparameter tuning and optimizer selection for model performance.
    - Integrating Google's API, using ``CONFIDENCE_MASK`` from [3] to refine skin detection and determine which parts of the image we are most confident about for classifying as skin. Experimenting with other pretrained ML models for skin segmentation on the market. Will use these segmentations as the training dataset to help with training the model and then compute results during inference time.
2. **Skin Tone Analysis:** After extracting the skin pixels, the second stage consists of determining the hue and tone of the extracted skin regions. After extracting skin pixels, we will analyze their color using established libraries like OpenCV to compute RGB values to determine the skin tone. Additionally, a CNN will be trained to predict continuous RGB values from the segmented regions, and its performance will be compared to the traditional color analysis methods for accuracy and robustness.

By combining advanced machine learning with established color analysis, we aim to build a robust tool that accurately detects skin tones from face images. This project could greatly improve how consumers choose cosmetics, offering personalized suggestions based on their skin tones.

### References
[1] Google AI. Image Segmentation Guide. Retrieved from https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter

[2] Ben Salah, K., Othmani, M., & Kherallah, M. (2021). A novel approach for human skin detection using convolutional neural network. *The Visual Computer, 38*(5), 1833–1843. doi:10.1007/s00371-021-02108-3

[3] Speelman, C. (2014). Skill Acquisition in Skin Cancer Detection Dataset. Retrieved from https://ro.ecu.edu.au/datasets/12/
