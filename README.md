# Smart-Speech-Analysis-Minor1
This repository contains the code of our minor project that we performed in my 5th semester.

In the repository files present are as follows:

--> previous models : This folder contains all the previosly trained models and the .ipynb files that contains the codes of their training. These are the models that are trained on some single dataset like TESS, RAVDESS, CREMA-D or our initial custom dataset which was the combinantion of these three datasets.

--> previous predictions : This folder contains the .csv files of predictions made by previos models on different datasets.

--> SER_by_NOR.ipynb : This is the jupyter nodetbbok file in which our final MLP model is trained on our final custom dataset.

--> SER_by_NOR.pkl : This is the our final trained MLP model file.

--> SRS.pdf : This is Software Requirement Specifiaction doc of our project.

--> main.py : This is the main python file our our GUI.

--> merge_datasets.ipynb : This is the jupyter file which contains the code of creating our final custom dataset.

--> prediction_custom_data_final.csv : This csv file contains the predictions made by our final model on the testing part of final custom dataset.

# Our Web App

Our webapp of this project cam be accessed through this link: https://soft-ui-pro-fkxy8h.teleporthq.app/

# Our Custom Dataset

This is a dataset that contains 19,487 Audio(.wav) files divided into 6 emotions {'happy','sad','angry','surprise','neutral','fear'}.

This dataset is a merged dataset that contains the audio files of above 6 emotions taken from the open source Speech Emotion Recognition datasets: RAVDESS, CREMA-D, TESS, SAVEE and ASVP-ESD.

You can easily acccess this dataset from Kaggle: 
