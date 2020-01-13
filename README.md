# SPEECH EMOTION PROBLEM 

This problem is based on the [MELD](https://github.com/SenticNet/MELD.git) dataset 
audio files were given for 5 classes of emotion -
1. happy
2. sad
3. neutral
4. disgust 
5. fear 

One had to completely extract files and make prepare the training and testing data in an organised format.
The work was done in the following manner -

### Data Preparation 
1. unzipping the file and retrieving all the folders with audio files
2. parsing through the folders to create a dataframe of labels 
3. parsing through the audio files, taking its label and file path and putting it in a dataframe for future utilization and ease 

### Data Analysis
1. Finding out the composition of the dataset, how many audios are of each category
2. Checking out the audios, hearing them in the Ipython display
3. Using wave library to analysis the graph of each audio file, zooming it and observing the details
4. Finding the length of the audio files and the frame rate
5. Comparing the length of audios of different categories 
6. Comparing the length of audios in the training and the validation data for each category 

### Data Preprocessing 
1. Finding that in order to make a predictive model how to convert the audio signal to numeric values
2. Extracting the MFCC features so as to convert the signal into numeric form, as a ML/DL model only takes numeric data 
3. Padding the dataset to ensure there is no unequal lengths 
4. splitting the data into features and labels for both training and validation data 
5. Normalizing the data in the X_train and X_test data
6. Converting the dataframes into numpy arrays 

### Prediction Model 

1. Deciding the right model ( Went with the deep learning approach as CNNs have been previosly used to work with Audio Data )
2. Reshaping the data for the models 
3. Building the network architectures of the model
4. Training the model on the training data
5. Evaluating the performance of the model 
6. checking the training of the model over time 
7. Deciding the better model, saving the weights and architecture 

#### Conv1D and Conv2D models were used, Conv2D performed better as 2D Conv Kernels were able to detect patterns slightly better 


## Conclusion
As the number of neutral samples were disproptionately high the model developed a bias towards predicting neutral 

Due to the limitation in time, this problem wasn't solved 
