import pandas as pd, numpy as np

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt
#Author : YongBaek Cho
#Description: This program will analyze the Arctic sea ice extent data and use my analysis to predict the date
#              the ice will disappear altogether

def get_Mar_Sept_frame():
    #Read data_79_17.csv
    df = pd.read_csv('data_79_17.csv', index_col = 0)
    df1 = df.loc[:,'0301':'0331'] # extract March value
    df2 = df.loc[:,'0901':'0931'] # extract September value
    aa = df1.mean(axis = 1) # March mean
    ab = df2.mean(axis = 1) # Sep mean
    a = np.ma.array(aa) 
    a1 = a.anom() #March Anomalies
    b = np.ma.array(ab)
    a2 = b.anom() #Sep Anomalies
    col = ['March_means', 'March_anomalies', 'September_means', 'September_anomalies']
    df3 = pd.DataFrame(index = df.index , columns = col)
    df3['March_means'] = aa
    df3['March_anomalies'] = a1
    df3['September_means'] = ab
    df3['September_anomalies'] = a2
    return df3
    
def get_ols_parameters(ts):
    # This function takes a Series, fits a line to it, and returns the slope, intercept, R squared, p- value.

    years_array = sm.add_constant(ts.index.values)
    model = sm.OLS(ts, years_array)
    results = model.fit()
    #print(results.summary())
    slope = float(results.params['x1'])
    R = float(results.rsquared)
    intercept = results.params['const']
    
    pval =  results.pvalues['x1']
    return slope, intercept, R, pval
    
    
def make_prediction(params, description='x-intercept:', x_name='x', y_name='y', ceiling=False):
    #This function print the prediction
    x_inter = -(params[1])/(params[0])
    if ceiling is True:
        x_inter = math.ceil(x_inter)
    print(description, x_inter)
    print("%0.f%%" % round(params[2] * 100),'of variation in ' + y_name + ' accounted for by ' +x_name +  ' (linear model)')
    print('Significance level of results:', '{:.1%}'.format(params[3]))
    if params[3] > 0.05:
        
        print('This result is not statistically significant.')
    else:
        print('This result is statistically significant.')
    
def make_fig_1(df):
    #This function takes a March-September frame
    pltDF = get_Mar_Sept_frame()
    plt.plot(pltDF.loc[:,'March_means'], linestyle = '-')
    
    plt.plot(pltDF.loc[:,'September_means'], linestyle = '-')
    ax = plt.gca()
    ax.set_ylabel(r"NH Sea Ice Extent ($10^6$ km$^2$)")
    ax.yaxis.label.set_fontsize(24)
    march_ols_params = get_ols_parameters(pd.Series(df['March_means'], df.index))
    xs_m = np.arange(1979, 2018)
    ys_m = march_ols_params[0] * xs_m + march_ols_params[1]
    plt.plot(xs_m, ys_m)
    September_ols_params = get_ols_parameters(pd.Series(df['September_means'], df.index))
    xs = np.arange(1979, 2018)
    ys = September_ols_params[0] * xs_m + September_ols_params[1]
    plt.plot(xs, ys)
    plt.autoscale(tight=True)
    
def make_fig_2(df2):
    #This function takes a March-September frame
    b = get_Mar_Sept_frame()    
    plt.plot(b.loc[:,'March_anomalies'], linestyle = '-')
    plt.plot(b.loc[:,'September_anomalies'], linestyle = '-')
    ax = plt.gca()
    ax.set_ylabel(r"NH Sea Ice Extent ($10^6$ km$^2$)")
    ax.yaxis.label.set_fontsize(24)
    ax.set_xlim(1979, 2017)
    ax.set_title('The Anomaly', fontsize=20)
    
    march_ols_params = get_ols_parameters(pd.Series(df2['March_anomalies'], df2.index))
    xs_m = np.arange(1979, 2018)
    ys_m = march_ols_params[0] * xs_m + march_ols_params[1]
    plt.plot(xs_m, ys_m)
    September_ols_params = get_ols_parameters(pd.Series(df2['September_anomalies'], df2.index))
    xs = np.arange(1979, 2018)
    ys = September_ols_params[0] * xs_m + September_ols_params[1]
    plt.plot(xs, ys)
    
def main():
    # Get the March-September frame. Get your OLS parameters for the four curves and printing a blank line.
    df = get_Mar_Sept_frame()
    
    a = pd.Series(df['March_means'], index = df['March_means'].index)
    b = pd.Series(df['September_means'], index = df['March_means'].index)
    c = get_ols_parameters(a)
    d = get_ols_parameters(b)
    make_prediction(c)
    make_prediction(d)
    make_fig_1(df)
    plt.figure()
    make_fig_2(df)
    plt.show()
main()
if __name__ == "__main__":
    main()