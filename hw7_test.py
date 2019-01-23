from hw7 import *
import hw7
import unittest, json, numpy as np, pandas as pd, io
from compare_pandas import *
from contextlib import redirect_stdout
import statsmodels.formula.api as smf
''' 
Auxiliary files needed:
    compare_pandas.py
    data_79_17.csv
    ms.pkl
'''

class TestFns(unittest.TestCase):
    def test_get_Mar_Sept_frame(self):
        correct = pd.read_pickle('ms.pkl')
        self.assertTrue(compare_frames(correct, get_Mar_Sept_frame(), 0.005))

    def test_get_ols_parameters(self):
        # UPDATE EACH YEAR:
        params = [
            [-0.04138431500587732, 98.01408098472064, 0.773757893391186, 1.686594805153962e-13],
            [-0.04138431500587739, 82.68586138174305, 0.7737578933911862, 1.6865948051538593e-13],
            [-0.08323027665317298, 172.4383072829542, 0.7608297824809106, 4.752326620841307e-13],
            [-0.08323027665317301, 166.29409275303973, 0.7608297824809106, 4.752326620841256e-13]
            ]
        March_September = pd.read_pickle('ms.pkl')
        #print('type', type(March_September['March_means']))
        #print(March_September['March_means'])
        self.assertTrue(compare_lists(params[0], get_ols_parameters(March_September['March_means'])))
        self.assertTrue(compare_lists(params[1], get_ols_parameters(March_September['March_anomalies'])))
        self.assertTrue(compare_lists(params[2], get_ols_parameters(March_September['September_means'])))
        self.assertTrue(compare_lists(params[3], get_ols_parameters(March_September['September_anomalies'])))
   
    def test_make_prediction(self):
        # UPDATE EACH YEAR with params[0][0] from test_get_ols_parameters (above):
        March_params = [-0.04138431500587732, 98.01408098472064, 0.773757893391186, 1.686594805153962e-13]
        #make_prediction(March_params, 'Ice-free March in year', 'time', 'Arctic sea ice', True)
        # UPDATE EACH YEAR with params[0][2] from test_get_ols_parameters (above):
        September_params = [-0.08323027665317298, 172.4383072829542, 0.7608297824809106, 4.752326620841307e-13]
        #make_prediction(September_params, 'Ice-free September in year', 'time', 'Arctic sea ice', True)
        # Using the F16 params - this year's prognostication is worse!  Already!
        # UPDATE EACH YEAR these two with blocks.  Run hw7.py and put in latest predictions:
        with io.StringIO() as buf, redirect_stdout(buf):
            correct = ("Ice-free March in year 2369\n" +
                "77% of variation in Arctic sea ice accounted for by time (linear model)\n" +
                "Significance level of results: 0.0%\n" +
                "This result is statistically significant.\n")
            self.assertIsNone(make_prediction(March_params, 'Ice-free March in year', 'time', 'Arctic sea ice', True))
            self.assertEqual(correct, buf.getvalue())
        with io.StringIO() as buf, redirect_stdout(buf): 
            correct = ("Ice-free September in year 2072\n" +
                "76% of variation in Arctic sea ice accounted for by time (linear model)\n" +
                "Significance level of results: 0.0%\n" +
                "This result is statistically significant.\n")
            self.assertIsNone(make_prediction(September_params, 'Ice-free September in year', 'time', 'Arctic sea ice', True))
            self.assertEqual(correct, buf.getvalue())
        # LEAVE THE REST ALONE:
        params = [-2.6, 10, 1.00, .050001]
        with io.StringIO() as buf, redirect_stdout(buf): 
            correct = ("x-intercept: 3.846153846153846\n" +
                "100% of variation in y accounted for by x (linear model)\n" +
                "Significance level of results: 5.0%\n" +
                "This result is not statistically significant.\n")
            self.assertIsNone(make_prediction(params))
            self.assertEqual(correct, buf.getvalue())
        with io.StringIO() as buf, redirect_stdout(buf): 
            correct = ("x-intercept: 4\n" +
                "100% of variation in y accounted for by x (linear model)\n" +
                "Significance level of results: 5.0%\n" +
                "This result is not statistically significant.\n")
            self.assertIsNone(make_prediction(params, ceiling=True))
            self.assertEqual(correct, buf.getvalue())
        params = [-2.5, 10, 1.00, .050000]
        with io.StringIO() as buf, redirect_stdout(buf): 
            correct = ("x-intercept: 4.0\n" +
                "100% of variation in y accounted for by x (linear model)\n" +
                "Significance level of results: 5.0%\n" +
                "This result is statistically significant.\n")
            self.assertIsNone(make_prediction(params))
            self.assertEqual(correct, buf.getvalue())
        
    
def main():
    test = unittest.defaultTestLoader.loadTestsFromTestCase(TestFns)
    results = unittest.TextTestRunner().run(test)
    print('Correctness score = ', str((results.testsRun - len(results.errors) - len(results.failures)) / results.testsRun * 60) + ' / 60')
    hw7.main()
    
if __name__ == "__main__":
    main()