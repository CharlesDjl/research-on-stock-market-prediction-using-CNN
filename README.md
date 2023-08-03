# research-on-stock-market-prediction-using-CNN
Bachelor Graduate Project

## Data Preparation
The experimental data in this study came from the api data interface provided by Tushare Big Data community (https://waditu.com/). The community has a wealth of data content, stocks, funds, futures, digital currency and other market data can be limited access, through the corresponding technology to the Shanghai Composite Index "four price" data in the form of csv table download backup.

The data set used in this study is based on Shanghai Composite Index data from January 4, 2010 to November 1, 2020
## Model Optimization

Factors: Observation window (Data Length), Kernel size, Kernel Numbers, model depth.

## RSI Introduction
RSI(Relative Strength Index) is a commonly used probabilistic financial technical indicator [13], created by Welles Wilder, and is currently a relatively commonly used short and medium term investment indicator in the technical analysis of the stock market. In essence, it is based on the principle of balance between supply and demand in the stock market, and horizontally compares the rise and fall of a certain stock or index in the same period of time to judge the strength of the strength of the buyer-seller's head, so as to predict the market trend.

## Result Comparison
1. Comparison between models with RSI and without RSI

   ![image](https://github.com/CharlesDjl/research-on-stock-market-prediction-using-CNN/assets/51400996/4358bacd-38d2-4897-8efb-4d7deddecaf5)
2. Prediction figure without RSI
   
![image](https://github.com/CharlesDjl/research-on-stock-market-prediction-using-CNN/assets/51400996/c2f20520-61a2-4a5b-afa9-31f6dedb3cc7)

3. Prediction figure with RSI

![image](https://github.com/CharlesDjl/research-on-stock-market-prediction-using-CNN/assets/51400996/4afafb0e-9a4d-4e6b-838f-cd76146f0908)

4. Comparison between traditional models

   ![image](https://github.com/CharlesDjl/research-on-stock-market-prediction-using-CNN/assets/51400996/24db40c2-d19d-4d10-89cc-9dd2ceb9f314)

## Conclusion
The financial technology index RSI is introduced and introduced into the old convolutional neural network model. Thanks to the judgment of RSI index on the trend of buyers and sellers, the lag of the model after the introduction of RSI is significantly improved, and the RSI-CNN stock index prediction model with excellent forecasting performance and practical value is finally obtained.

In general, this study is based on the theory of "quadrivalence" in Japanese candle graph technology, essentially transforming "quadrivalence" into image-like matrix data and analyzing its automatic feature extraction with convolutional neural network model. The innovation of this study is to introduce the RSI index into the stock index prediction model to improve the degree of prediction fitting and the original model
