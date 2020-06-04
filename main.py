"""
Author: Jason Wherry	Start Date: 5/08/2020		To Run: python3 main.py

What am I actually plotting? --> The means calculated from each sample

Next steps:
	- Create a Menu for the user
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
	case_sizes = ['1', '40', '80', '160', '320', '640']
	cases = input('Enter \'q\' to quit\nChoose the number of sample(s) – 1  40  80  160  320  640: ')

	# check input to stop code execution
	if cases == 'q' or cases == 'Q':
		exit()

	while cases not in case_sizes:
		print(cases, 'is an invalid option')
		cases = input('\nEnter \'q\' to quit\nChoose the number of sample(s) – 1  40  80  160  320  640: ')
		
		# check input to stop code execution
		if cases == 'q' or cases == 'Q':
			exit()

	samples = int(population / int(cases) )
	print('\n')

	# empty data frame to hold the data; each column is a case
	df = pd.DataFrame(data=None)

	# fill the DataFrame with data;		df.shape = [samples X cases]
	# each column is a sample case with the number of values equal to the ratio (population/cases)
	for i in range(0, int(cases) ):
		temp = pd.Series(np.random.uniform(0, 1.0, samples))
		col_name = 'Case ' + str(i+1)
		df.insert(i, col_name, temp)

	print(df, '\n') # display np.random.uniform numbers (dtype is float64) by sample

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

		# Turn array of means into a numpy array
		numpy_means = np.array(means)

		fig = plt.subplot()
		fig.hist(numpy_means, bins=50, range=[0,1], histtype='bar')
		fig.set_xlabel('Mean')
		fig.set_ylabel('Frequency')
		plt.show()


	else:
		# Turn array of means into a numpy array
		numpy_means = np.array(means)

		fig = plt.subplot()
		fig.hist(numpy_means, bins=50, range=[0,1], histtype='bar')
		fig.set_xlabel('Mean')
		fig.set_ylabel('Frequency')
		plt.show()

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
	# for i in range(0, len(means)):
	# 	print('\tsample number', i+1, '\tmean\t', means[i])

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
 	- objective:		runs the program functions
	- parameter:	 	none
	- return:			none
"""
def test():
	run = prompt_user() # returns a data frame
	means, mean_of_means, std_dev, skew, kurt = find_stats(run)
	display_stats(means, mean_of_means, std_dev, skew, kurt)


test() # run 1
test() # run 2
test() # run 3
test() # run 4
test() # run 5
test() # run 6

