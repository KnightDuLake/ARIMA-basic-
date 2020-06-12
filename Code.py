# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:01:10 2020

@author: KightDuLake
"""

# Load library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA



#ADF test
def adfTest(data):
    rolmean = data.rolling(7).mean()
    rol_weighted_mean = data.ewm(span = 7).mean()
    rolstd = data.rolling(7).std()
    dftest = adfuller(data, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
    
    Orig = plt.plot(data, color = 'blue', label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    weighted_mean = plt.plot(rol_weighted_mean, color = 'green', label = 'Weighted Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Std')
    plt.show(block = False)
    
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print(dfoutput)
    return dfoutput

#Training Set
def Training(after,revenue_all, end):
    revenue =  revenue_all.iloc[after:end]
    revenue.dropna(inplace = True)
    revenue = revenue.reset_index(drop=True)
    return revenue

 #Testing Set 
def Testing(after,revenue_all, end, testing_len):
    revenue_testing =  revenue_all.iloc[end:end+testing_len]
    revenue_testing.dropna(inplace = True)
    Testrev = revenue_testing.reset_index(drop=True)
    return Testrev


def Stability(revenue):
    #Show p-value for 1/2 order difference
    rev_log = np.log(revenue)
    rev_log_diff1 = rev_log.diff(1)
    rev_log_diff1.dropna(inplace=True)
    rev_log_diff2 = rev_log_diff1.diff(1)
    rev_log_diff2.dropna(inplace=True)
    adfD1 = adfTest(rev_log_diff1)
    adfD2 = adfTest(rev_log_diff2)
    return adfD1['p-value'], adfD2['p-value']
    
def ARIMA_model(revenue,d):
    rev_log = np.log(revenue)
    para = sts.arma_order_select_ic(rev_log,ic='aic')['aic_min_order'] #AIC原则选参数
    model = ARIMA(rev_log, order=(para[0], d, para[1]) )    
    model_fit = model.fit()
    
    Pred_log = rev_log[0:d]
    Pred_log = Pred_log.append(model_fit.predict(typ = 'levels'))
    Pred_log = Pred_log.reset_index(drop = True)
    
    plt.plot(rev_log, color = 'blue', label = 'Log Original')
    plt.plot(Pred_log, color='red',label = 'Predicted Training Log')
    plt.legend(loc = 'best')
    plt.show()  
    
    return model_fit

def ARIMA_TrainValue(revenue, model_fit,d):
    rev_log = np.log(revenue)
    
    Pred_log = rev_log[0:d]
    Pred_log = Pred_log.append(model_fit.predict(typ = 'levels'))
    Pred_log = Pred_log.reset_index(drop = True)
    
    predictions = np.exp(Pred_log)
    
    plt.plot(revenue)
    plt.plot(predictions)
    plt.title('predictions_ARIMA RMSE: %.4f' % np.sqrt(sum((predictions - revenue) ** 2) / len(revenue)))
    plt.show()

    cumu_predicts = np.cumsum(predictions)
    cumu_real = np.cumsum(revenue)
    
    plt.plot(cumu_predicts, label='predicts')
    plt.plot(cumu_real, label=('real'))
    plt.legend(loc= 'best')
    plt.title('predictions_ARIMA Total Difference: %s' % round((cumu_predicts[len(cumu_predicts) -1] - cumu_real[len(cumu_real) -1])/cumu_real[len(cumu_real) -1],2))
    plt.show()
    
    return predictions

def ARIMA_Forward(revenue, testing_len,model_fit):
    rev_log = np.log(revenue)
    start_index = len(rev_log)
    end_index = start_index + testing_len -1 
    forecast = model_fit.predict(start=start_index, end=end_index,typ='levels')
    forecast = forecast.reset_index(drop=True)
    
    Test_Forecast = np.exp(forecast)
    plt.plot(Test_Forecast, label = 'Forecast')
    plt.legend(loc = 'best')
    plt.show()

    cumu_predicts_test = np.cumsum(Test_Forecast)
    plt.plot(cumu_predicts_test, label='predicts')
    plt.legend(loc= 'best')
    plt.show()
    
    return Test_Forecast

def ARIMA_TestValue(test_revenue,revenue,testing_len,model_fit):
    rev_log = np.log(revenue)
    start_index = len(rev_log)
    end_index = start_index + testing_len -1 
    forecast = model_fit.predict(start=start_index, end=end_index,typ='levels')
    forecast = forecast.reset_index(drop=True)
    
    Test_Forecast = np.exp(forecast)
    plt.plot(Test_Forecast, label = 'Forecast')
    plt.plot(test_revenue, label = 'Testing Set')
    plt.legend(loc = 'best')
    plt.show()

    cumu_predicts_test = np.cumsum(Test_Forecast)
    cumu_real_test = np.cumsum(test_revenue)
    plt.plot(cumu_predicts_test, label='predicts')
    plt.plot(cumu_real_test, label=('real'))
    plt.title('predictions_ARIMA Total Difference: %s' % round((cumu_predicts_test[len(cumu_predicts_test) -1] - cumu_real_test[len(cumu_real_test) -1])/cumu_real_test[len(cumu_real_test) -1],2))
    plt.legend(loc= 'best')
    plt.show()
    
    return Test_Forecast


if __name__ == '__main__':
    #Read data
    data = pd.read_excel(r'C:\Users\KightDuLake\Desktop\test.xlsx')
    revenue_all = data['OACO']
    after = 15
    end = 80  #decide training set size
    testing_len = 20 # decide testing set
    forward_step = 20 # decide predicting step (only used in soley forward predict)
    
    revenue = Training(after,revenue_all, end) # Training set
    test_revenue = Testing(after,revenue_all, end, testing_len) # Testing set
    diff_p = Stability(revenue) #show adf test result for 1-order difference and 2-order difference
    
    d = 1 #Compare result in diff_p, if first one is already small enough (representing p-value), then d = 1, else d = 2
    
    model_fit = ARIMA_model(revenue, d) #Build ARIMA model, using AIC selection parameters automatically
    Train_predict = ARIMA_TrainValue(revenue, model_fit,d) #Review the result in Training Set
    Test_predict = ARIMA_TestValue(test_revenue,revenue, testing_len, model_fit) #Review the effect in Testing Set
    Forward_predict = ARIMA_Forward(revenue,forward_step,model_fit) # Solely predict forward


