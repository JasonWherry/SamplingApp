"""
Author: Jason Wherry	Start Date: 5/08/2020		To Run: python3 main.py

What am I actually plotting? --> The means calculated from each sample

Next steps:
	- Add on to educate()
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
	- parameter:	 	None
	- return:			DataFrame – filled with values randomly generated between 0 & 1
"""
def prompt_user(distribution):
	population = input('\nEnter \'q\' to quit\nChoose a Population size - 3200  6400  9600, default=3200: ') #3200
	population_sizes = ['3200', '6400', '9600']
	
	# check input to stop code execution
	if population == 'q' or population == 'Q':
		exit()
	if population == '':
		population = '3200'

	while population not in population_sizes:
		print(population, ' - invalid population')
		population = input('\nEnter \'q\' to quit\nChoose a Population size - 3200  6400  9600, default=3200: ')
		
		# check input to stop code execution
		if population == 'q' or population == 'Q':
			exit()
		if population == '':
			population = '3200'

	print('\npopulation =', population)

	case_sizes = ['1', '40', '80', '160', '320', '640']
	cases = input('\n\tEnter \'q\' to quit\n\n\tChoose the number of cases – 1  40  80  160  320  640: ')

	# check input to stop code execution
	if cases == 'q' or cases == 'Q':
		exit()

	while cases not in case_sizes:
		print(cases, ' - invalid case')
		cases = input('\nEnter \'q\' to quit\nChoose the number of cases – 1  40  80  160  320  640: ')
		
		# check input to stop code execution
		if cases == 'q' or cases == 'Q':
			exit()

	print('cases =', cases)

	population_int = int(population)

	samples = int(population_int / int(cases) )
	print('samples =', str(samples) )
	print('\n')

	# empty data frame to hold the data; each column is a case
	df = pd.DataFrame(data=None)

	# fill the DataFrame with data;		df.shape = [samples X cases]
	# each column is a sample case with the number of values equal to the ratio (population/cases)

	if distribution == 'Uniform':
		for i in range(0, int(cases) ):
			temp = pd.Series(np.random.uniform(0, 1.0, samples))
			col_name = 'Case ' + str(i+1)
			df.insert(i, col_name, temp)

	# elif distribution == 'Bernoulli':
		# for i in range(0, int(cases) ):
		# 	temp = pd.Series(np.random.binomial(1, 0.5, samples))
		# 	col_name = 'Case ' + str(i+1)
		# 	df.insert(i, col_name, temp)
		# # Not working at the moment

	elif distribution == 'Binomial':
		for i in range(0, int(cases) ):
			temp = pd.Series(np.random.binomial(1, 0.5, samples)) # flip a coin 1 time, tested samples times
			col_name = 'Case ' + str(i+1)
			df.insert(i, col_name, temp)

	elif distribution == 'Normal':
		for i in range(0, int(cases) ):
			temp = pd.Series(np.random.normal(0.5, 0.1, samples))
			col_name = 'Case ' + str(i+1)
			df.insert(i, col_name, temp)

	print(df, '\n') # display np.random.uniform numbers (dtype is float64) by sample
	# print(df.shape)

	return df, distribution


"""
 find_mean(data_frame)
 	- objective:		determine the mean of the chosen sample(s)
	- parameter:	 	pandas DataFrame, type of distribution
	- return:			Series – means & std. deviations
"""
def find_stats(df, dist_type):
	means = []
	sample_num = len(df.columns) 	# AKA cases
	sample_size = len(df)			# AKA samples per case
	dist_type = dist_type + ' Distribution - ' + str(sample_num) + ' Samples that observe ' + str(sample_size) + ' numbers'

	print('number of cases:\t', sample_num, '\nsamples per case:\t', sample_size)
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
		fig.set_xlabel('Mean (Value)')
		fig.set_ylabel('Value Frequency')
		fig.set_title(dist_type)
		plt.show()


	else:
		# Turn array of means into a numpy array
		numpy_means = np.array(means)

		fig = plt.subplot()
		fig.hist(numpy_means, bins=50, range=[0,1], histtype='bar')
		fig.set_xlabel('Mean (Value)')
		fig.set_ylabel('Value Frequency')
		fig.set_title(dist_type)
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
	- return:			None
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
	- parameter:	 	number of runs desired, selected distribution
	- return:			None
"""
def test(num_runs, dist):
	while num_runs > 0:
		run, dist = prompt_user(dist) # returns a data frame
		means, mean_of_means, std_dev, skew, kurt = find_stats(run, dist)
		display_stats(means, mean_of_means, std_dev, skew, kurt)

		num_runs -= 1


# test(1, 'Normal') # run once

"""
educate()
	- objective:		Provide user with a list of options to run the program
	- parameter:		None
	- return:			User's request
"""
def educate():
	print('A Uniform distribution...')

	# print('A Bernoulli distribution...')

	print('A Normal distribution...')

	print('A Binomial distribution...')


"""
 menu()
	- objective:		Provide user with a list of options to run the program
	- parameter:		None
	- return:			User's request
"""
def menu():
	print('\n  Menu')
	print('\t1 : distributions -> lists alternative distribution choices')
	print('\t2 : demonstration -> demonstrates the Central Limit Theorem with random sampling')
	print('\t3 : quit -> terminates program execution')

	user_input = input('\n  Enter one of the following options: \'1\', \'2\', \'3\': ')

	# print(user_input)
	print('\n')

	# case analysis on user input
	if user_input == '1':
		educate()
		menu() # display menu again
	elif user_input == '2':
		print('\n Select a distribution type')
		print('\t1 : Uniform')
		# print('\t2 : Bernoulli')
		print('\t2 : Binomial')
		print('\t3 : Normal')
		

		dist_type = input('\n  Enter one of the following options: \'1\', \'2\', \'3\': ')
		


		if dist_type == '1':
			test(1, 'Uniform')
		# elif dist_type == '2':
		# 	test(1, 'Bernoulli')
		elif dist_type == '2':	
			test(1, 'Binomial')
		elif dist_type == '3':
			test(1, 'Normal')
		

		menu() # display menu again
	elif user_input == '3':
		exit()
	

menu()





