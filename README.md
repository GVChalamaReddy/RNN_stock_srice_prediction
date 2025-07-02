# Stock Price Prediction Using RNNs
**Problem Statement:** Given the stock prices of Amazon, Google, IBM, and Microsoft for a set number of days, predict the stock price of these companies after that window.

## Table of Contents
* [General Info](#general-information)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
  ### Business Objective: 
  The objective of this assignment is to try and predict the stock prices using historical data from four companies IBM (IBM), Google (GOOGL), Amazon (AMZN), and Microsoft (MSFT).
  We use four different companies because they belong to the same sector: Technology. Using data from all four companies may improve the performance of the model. This way, we can capture the broader market sentiment.

  Data related to stock markets lends itself well to modeling using RNNs due to its sequential nature. We can keep track of opening prices, closing prices, highest prices, and so on for a long period of time as these values are generated every working day. The patterns observed in this data can then be used to predict the future direction in which stock prices are expected to move. Analyzing this data can be interesting in itself, but it also has a financial incentive as accurate predictions can lead to massive profits.

  ### Data Description:
  You have been provided with four CSV files corresponding to four stocks: AMZN, GOOGL, IBM, and MSFT. The files contain historical data that were gathered from the websites of the stock markets where these companies are listed: NYSE and NASDAQ. The columns in all four files are identical. Let's take a look at them:
    
  - `Date`: The values in this column specify the date on which the values were recorded. In all four files, the dates range from Jaunary 1, 2006 to January 1, 2018.
  - `Open`: The values in this column specify the stock price on a given date when the stock market opens.
  - `High`: The values in this column specify the highest stock price achieved by a stock on a given date.
  - `Low`: The values in this column specify the lowest stock price achieved by a stock on a given date.
  - `Close`: The values in this column specify the stock price on a given date when the stock market closes.
  - `Volume`: The values in this column specify the total number of shares traded on a given date.
  - `Name`: This column gives the official name of the stock as used in the stock market.

  ### Approach:
  - All the images were the size of 256×256. Resize of images is done without loosing the data, inorder to reduce the compute memory issue .
  - All the images have 3 channels as RGB.
  - The data has seven categories. Cardboard, Food Waste, Glass, Metal, Other, Paper, Plastic
  - Class distribution was imbalanced. There was a greater number of images in the plastic category.
  - Since there was no equal number of images present in each category, the test and train split was done with stratify=y with an 80/20 ratio.

 ### Techniques:
 Specific tools or methods used at different steps of model building are mentioned below: 
 -	**Data preprocessing**: One-Hot encoding
 -	**Model building**: Simple RNN Network, LSTM
 -	**Model Optimization**: Dropout, Regularization, Learning Rate, Batch Normalization, Adam Optimizer
 -	**Evaluation metrics**: Accuracy, Confusion Matrix, Recall, Precision, F1-score

## Conclusions
- Started building own CNN model with 3 convolutional layers, but the accuracy was just 0.40, which was very low. I tried configuring hyperparameters like dropouts, and the performance increased to 0.50.
- As own CNN model is overfitting, used RESNET 50 with transfer learning approach.Initially made all the layers trainable as false observed only 0.60 training accuracy, a slight increase from own CNN model.
- Modified the trainable layers to 20 and got the training accuracy as 98 percent and the validation accuracy as 60%. Clearly the model was overfitting.
- To resolve overfitting problem, modified the trainable layers from 20 to 10, and still able to see the model is overfitting.
- Modified model training layer to 5 and increased the dropout from 0.6 to 0.7 and reduced the learning rate. with these configuration training accuracy: 0.86 and val_accuracy: 0.80 are improved signficantly.
- Clearly, the model is able to generalize the training data and produce the desired output.
- From the confusion matrix we are able to see a lot of misclassification is done as a plastic when compared to others, which is very little.
- Overall, the model was able to generalize the data and produce the acceptable prediction; nearly 86% of the data is properly classified.
- The model has performed well on the waste segregation task, achieving overall 84% accuracy with strong generalization.


## Technologies Used
- **python** - 3.13.1
- **numpy** - 2.2.1
- **pandas** - 2.2.3
- **matplotlib** - 3.10.0
- **seaborn** - 0.13.2
- **statsmodels** - 0.14.4
- **sklearn** - 1.6.1
- **keras** - 3.8.0
- **PIL** - 11.1.0
- **tensorflow** - 2.18.0

## Acknowledgements

- This project was inspired by Upgrad IIIT Bangalore PG program on ML and AI.
- This project was based on Convolutional Neural Networks (CNNs)


## Contact
Created by @[GVChalamaReddy](https://github.com/GVChalamaReddy) - feel free to contact me!
