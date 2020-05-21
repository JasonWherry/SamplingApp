"""
Author: Jason Wherry	Start Date: 5/08/2020		To Run: python3 main.py

What am I actually plotting?

"""

import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sb
import matplotlib.pyplot as plt
from statistics import stdev


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
		std_dev = round( df.iloc[0:, i].mean().std(), 3)

	else:
		# calculates the std. deviation of each sample's mean
		std_dev = round( stdev(means), 3)

	return means, std_dev


"""
 find_mean(data_frame)
 	- objective:		display content of lists
	- parameter:	 	pandas Series
	- return:			none
"""
def display_stats(means, std_dev):
	for i in range(0, len(means)):
		print('\tsample number', i+1, '\tmean', means[i])

	print('\n')
	print('\t\t\t\tstd. deviation', std_dev)
	print('\n\n')


"""
 test()
 	- objective:		runs the program functions five separate times
	- parameter:	 	none
	- return:			none
"""
def test():
	run_1 = prompt_user()
	means_1, std_dev_1 = find_stats(run_1)
	display_stats(means_1, std_dev_1)

	run_2 = prompt_user()
	means_2, std_dev_2 = find_stats(run_2)
	display_stats(means_2, std_dev_2)

	run_3 = prompt_user()
	means_3, std_dev_3 = find_stats(run_3)
	display_stats(means_3, std_dev_3)

	run_4 = prompt_user()
	means_4, std_dev_4 = find_stats(run_4)
	display_stats(means_4, std_dev_4)

	run_5 = prompt_user()
	means_5, std_dev_5 = find_stats(run_5)
	display_stats(means_5, std_dev_5)

	run_6 = prompt_user()
	means_6, std_dev_6 = find_stats(run_6)
	display_stats(means_6, std_dev_6)

test()


