"""
Author: Jason Wherry	Start Date: 5/08/2020		To Run: python3 main.py

What am I actually plotting?

"""

import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sb
import matplotlib.pyplot as plt
from statistics import *
import scipy.stats as stats


"""
 prompt_user()
 	- objective:		gather data aligned with user's request
	- parameter:	 	none
	- return:			DataFrame – filled with values randomly generated between 0 & 1
"""
def prompt_user():
	population = 3200
	case_sizes = [1, 40, 80, 160, 320, 640]
	cases = input('Choose the number of sample(s) – 1  40  80  160  320  640: ')
	cases = int(cases)

	while cases not in case_sizes:
		print(cases, 'is an invalid option')
		cases = input('Choose the number of sample(s) – 1  40  80  160  320  640: ')
		cases = int(cases)

	samples = int(population / cases)
	print('\n')

	# empty data frame to hold the data; each column is a case
	df = pd.DataFrame(data=None)

	# fill the DataFrame with data;		df.shape = [samples X cases]
	# each column is a sample case with the number of values equal to the ratio (population/cases)
	for i in range(0, cases):
		temp = pd.Series(np.random.uniform(0, 1.0, samples))
		col_name = 'Case ' + str(i+1)
		df.insert(i, col_name, temp)

	# print(df, '\n') # display np.random.uniform numbers (dtype is float64) by sample

	return df


"""
 find_mean(data_frame)
 	- objective:		determine the mean of the chosen sample(s)
	- parameter:	 	pandas DataFrame
	- return:			Series – means & std. deviations
"""
def find_stats(df):
	means = []
	sample_num = len(df.columns)
	sample_size = len(df)

	print('number of sample(s):\t', sample_num, '\nsize of each sample:\t', sample_size)
	print('\n')
	
	# loop through each sample and find its respective mean
	for i in range(0, sample_num):
		mean = round( df.iloc[0:, i].mean(), 3) # mean of sample size from generated random.uniform values
		means.append(mean)

	if sample_num == 1:
		# there is one sample & we need more than 1 value to calculate std_dev
		# std_dev = round( df.iloc[0:, i].std(), 3) # old way to calculate std_dev for just 1 sample
		mean_of_means = means[0]
		std_dev = round( df.iloc[0:, i].mean().std(), 3)
		skewness = round( df.iloc[0:, i].skew(), 3)
		kurtosis = round( df.iloc[0:, i].kurtosis(), 3)

	else:
		# calculates the std. deviation of each sample's mean
		numpy_means = np.array(means)
		mean_of_means = round( numpy_means.mean(), 3)
		std_dev = round( stdev(means), 3)
		skewness = round( stats.skew(means), 3)
		kurtosis = round( stats.kurtosis(means), 3)


	return means, mean_of_means, std_dev, skewness, kurtosis


"""
 find_mean(data_frame)
 	- objective:		display content of lists
	- parameter:	 	pandas Series
	- return:			none
"""
def display_stats(means, mean_of_means, std_dev, skewness, kurtosis):
	for i in range(0, len(means)):
		print('\tsample number', i+1, '\tmean\t', means[i])

	print('\n')
	# print('\t\t\t\t|–––––––––––––––––––––––––––––––|')
	print('\t\t\t|–––––––––––––––––––––––|')
	print('\t\t\t  mean of means ', mean_of_means)
	print('\t\t\t  std. deviation', std_dev)
	print('\t\t\t  skewness\t', skewness)
	print('\t\t\t  kurtosis\t', kurtosis)
	# print('\t\t\t\t|\t\t\t\t|')
	print('\t\t\t|–––––––––––––––––––––––|')
	print('\n\n')


"""
 test()
 	- objective:		runs the program functions five separate times
	- parameter:	 	none
	- return:			none
"""
def test():
	run_1 = prompt_user()
	means_1, mean_of_means_1, std_dev_1, skew_1, kurt_1 = find_stats(run_1)
	display_stats(means_1, mean_of_means_1, std_dev_1, skew_1, kurt_1)

	run_2 = prompt_user()
	means_2, mean_of_means_2, std_dev_2, skew_2, kurt_2 = find_stats(run_2)
	display_stats(means_2, mean_of_means_2, std_dev_2, skew_2, kurt_2)

	run_3 = prompt_user()
	means_3, mean_of_means_3, std_dev_3, skew_3, kurt_3 = find_stats(run_3)
	display_stats(means_3, mean_of_means_3, std_dev_3, skew_3, kurt_3)

	run_4 = prompt_user()
	means_4, mean_of_means_4, std_dev_4, skew_4, kurt_4 = find_stats(run_4)
	display_stats(means_4, mean_of_means_4, std_dev_4, skew_4, kurt_4)

	run_5 = prompt_user()
	means_5, mean_of_means_5, std_dev_5, skew_5, kurt_5 = find_stats(run_5)
	display_stats(means_5, mean_of_means_5, std_dev_5, skew_5, kurt_5)

	run_6 = prompt_user()
	means_6, mean_of_means_6, std_dev_6, skew_6, kurt_6 = find_stats(run_6)
	display_stats(means_6, mean_of_means_6, std_dev_6, skew_6, kurt_6)

test()


