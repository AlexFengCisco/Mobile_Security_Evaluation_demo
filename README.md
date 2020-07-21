### Tensorflow sample code for Mobile security evaluation 

#### Requirements 
    python 3.7
    tensorflow 1.13.1
    
#### Sample dataset

    Total collections divied into two sets, training and test , 75% for training 20% for testing.
    
    The first line is a header containing information about the dataset 
    
    There are 120 total collections. Each  has 6 features and one of three possible label names
    1 2 3 ... for mobile security score , menas 1 for 10% ,2 for 20% etc.
    
    Subsequent rows are data collections , Mobile handset quality , Location , Apps quality etc , and marked with security score pecentage.
    
#### Rraining 

    Build 3 hidden layers DNN with 10, 20, 10 units respectively. n_classes is the number of classes , for 100% score  , n_classes = 10 for 10% 20% 30% ... 100%
    
    Load training dataset , back propagation to build graph, sample code with 2000 steps
    
    load testing dataset to evaluate accuracy.Test Accuracy: ~0.966667
    
    tensorflow graph model saved.
    
#### Classify Prediction

     Use saved tensorflow graph model to classify input mobile data , and output security score.


#### Remove saved graph model before modify DNN and dataset