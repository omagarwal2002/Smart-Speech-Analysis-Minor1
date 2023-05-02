# Smart-Speech-Analysis-Minor1
Automatic speech recognition is basically the process of converting spoken words into text form, basically transcribing what someone is speaking. It is a challenging problem to solve, but you can see various examples of this technology at work nowadays. Since emotions help us to understand each other better, a natural outcome is to extend the understanding to computers. Speech recognition is already in our everyday life, thanks to smart mobile devices that are able to accept and reply to synthesized speech. Speech emotion recognition could be used to enable them to detect our emotions as well.

If you have an android phone and just say ‘OK Google”, you will see a window open up at the bottom. If you now speak more words, you will see that the app will try to identify what you are speaking. It won't get perfect the first time, and it probably will show some intermediate words as you continue to speak. But, in the end, the technology will recognize your speech.

# About this repo
This repository contains the code of our minor project that we performed in my 5th semester.

In the repository files present are as follows:

--> previous models : This folder contains all the previosly trained models and the .ipynb files that contains the codes of their training. These are the models that are trained on some single dataset like TESS, RAVDESS, CREMA-D or our initial custom dataset which was the combinantion of these three datasets.

--> previous predictions : This folder contains the .csv files of predictions made by previos models on different datasets.

--> Project Presentation.pptx : It is the ppt of our final presenatation of this project. 

--> SER_by_NOR.ipynb : This is the jupyter nodetbbok file in which our final MLP model is trained on our final custom dataset.

--> SER_by_NOR.pkl : This is the our final trained MLP model file.

--> SRS.pdf : This is Software Requirement Specifiaction doc of our project.

--> main.py : This is the main python file our our GUI.

--> merge_datasets.ipynb : This is the jupyter file which contains the code of creating our final custom dataset.

--> prediction_custom_data_final.csv : This csv file contains the predictions made by our final model on the testing part of final custom dataset.

# Our Web App

Our webapp of this project cam be accessed through this link: https://soft-ui-pro-fkxy8h.teleporthq.app/

< Currently we are working on it. > 

# Our Custom Dataset

This is a dataset that contains 19,487 Audio(.wav) files divided into 6 emotions {'happy', 'sad', 'angry', 'surprise', 'neutral', 'fear'}.

This dataset is a merged dataset that contains the audio files of above 6 emotions taken from the open source Speech Emotion Recognition datasets: RAVDESS, CREMA-D, TESS, SAVEE and ASVP-ESD.

You can easily acccess this dataset from Kaggle: https://www.kaggle.com/datasets/omagarwal2411/nor-smart-speech?datasetId=2716311&sortBy=dateRun&tab=profile
