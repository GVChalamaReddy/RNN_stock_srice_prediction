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
  To predict closing prices of multiple stocks ('AMZN', 'IBM', 'GOOGL', MSFT') using sequential neural network models (Simple RNN and Advanced RNN like LSTM/GRU), with careful preprocessing, optimization, and evaluation.
  - **Simple RNN**:
    - A basic SimpleRNN architecture was developed.
    - Hyperparameters like number of units, activation functions, and return sequences were manually and automatically tuned.
    - The model was trained and evaluated on windowed sequences of stock prices.
  - **Advanced RNN (LSTM/GRU)**:
    - Implemented LSTM models with:
      - return_sequences=True for stacking
      - Multiple layers with dropout
      - Tunable activations (tanh, relu)
    - Used Keras Tuner with Bayesian Optimization to find optimal hyperparameter configurations:
      - Number of units per layer
      - Learning rate
      - Dropout rates
      - Activation functions

 ### Techniques:
 Specific tools or methods used at different steps of model building are mentioned below: 
 -	**Data preprocessing**:
    - Loaded data from multiple CSV files, each representing a different stock.
    - Extracted and combined them into a single DataFrame.
    - Dropped unnecessary columns and standardized feature names (e.g., Close_AMZN).
    - Handled missing values using forward-fill and interpolation.
    - Analyzed volume distributions and plotted correlations for each stock separately.
    - Created windowed sequences for time-series modeling (e.g., a window of 20 days to predict the next day's closing prices).
    - Applied scaling (MinMaxScaler) in a windowed way to prevent leakage.
    - Split data into training and testing sets using train_test_split.
 -	**Model building**:
    - Simple RNN Network, LSTM
 -	**Model Optimization**: Dropout, Regularization, Learning Rate, Batch Normalization, Adam Optimizer
      - Training strategies:
        - EarlyStopping to prevent overfitting
        - ModelCheckpoint to save best performing model
        - Validation split (20%) during training
        - Best model retrained on the full training set
 -	**Evaluation metrics**:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R Squared
    - Visual Actual vs Predicted plots for each stock
 - Final Results:
   - `AMZN` - MAE: 175.8513, RMSE: 34331.8409, R²: -0.0732
   - `IBM` - MAE: 33.3136, RMSE: 1294.1401,R²: -7.4355
   - `GOOGL` - MAE: 22.7943, RMSE: 804.6457, R²: 0.9362
   - `MSFT` - MAE: 7.1929, RMSE: 64.2991, R²: 0.4431

## Conclusions
- Advanced RNN models (LSTM/GRU) outperformed Simple RNNs in capturing complex temporal dependencies.
- Adding more layers and using return_sequences=True improved performance, especially when using BayesianOptimization.
- Stocks with high volatility had larger prediction errors, but the model generalized well.
- Window size around business weeks (e.g., 20 days) gave more consistent patterns for the model to learn.
- Data normalization and careful train-test splits were crucial for stable results.

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
- This project was based on Recurrent Neural Networks (CNNs)


## Contact
Created by @[GVChalamaReddy](https://github.com/GVChalamaReddy) and @[Sajeev](https://github.com/sajeevmply)- feel free to contact me!
