# Thesis
Master Thesis: Video Description Generation using Multimodal Models
Aim: The automatic video description aims to generate natural language sentences that contain a short summary 
of the video content by highlighting the important features and events from the video. A highly efficient 
automatic video description can improve the human-machine interface. The accuracy of currently available methods 
for automatic video description still needs improvement compared to the performance of this task by a human. 
In this project, a model that can automatically describe the content of the video by analyzing two different 
modalities, such as video and audio is introduced. This is done by implementing unimodal and multimodal models 
based on Sequence-to-Sequence and attention mechanism structures. The training of the models is done on the
MSR-VTT dataset. Visual features are obtained using a pre-trained model for image classification. 
Audio features are represented by Mel Frequency Cepstral Coefficient (MFCC) data. 

1. Video feature extraction using VGG16
2. Text processing
3. Sequence-to-Sequence model

Dataset description
Simulations are conducted using the MSR-VTT dataset. This dataset is developed by Microsoft Research and 
consists of 10K video clips of 20 different categories where each video is described with 20 sentences.
The main advantages of this dataset over others are the diversity of video, the complexity of video content 
and the availability of audio channels. The dataset is divided into training (6513 videos), validation (457 videos)
and test (2990) subsets.
