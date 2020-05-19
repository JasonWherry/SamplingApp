"""
Author: Jason Wherry	Start Date: 5/08/2020		To Run: python3 main.py
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import norm
import seaborn as sb
import matplotlib.pyplot as plt

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

# fill the DataFrame with data
# df.shape = [samples X cases]
# each column is a sample case with the number of values equal to the ratio (population/cases)
for i in range(0, cases):
	temp = pd.Series(np.random.uniform(0, 1.0, samples))
	col_name = 'Case ' + str(i+1)
	df.insert(i, col_name, temp)

print(df, '\n')


"""
 find_mean(data_frame)
 	- objective:		determine the mean of the chosen sample(s)
	- parameter:	 	pandas DataFrame
	- return:			none
"""
def find_mean(data_frame):
	sample_num = len(df.columns)
	sample_size = len(df)

	print('number of sample(s):\t', sample_num, '\nsize of each sample:\t', sample_size)
	print('\n')
	
	# loop through each sample and find the mean?
	for i in range(0, sample_num):
		mean = round( df.iloc[0:, i].mean(), 3)
		std_dev = round( df.iloc[0:, i].std(), 3)
		print('sample number', i+1, '\tmean', mean, '\tstd. deviation', std_dev)

find_mean(df)



# Visualize (plot) the chosen sample's mean
'''
low_bound = 0
high_bound = 1

z1 = (low_bound - mean) / std_dev
z2 = (high_bound - mean) / std_dev

x = np.arange(z1, z2, 0.01) # range of x in spec
x_all = np.arange(-1, 1, 0.01) # entire range of x, both in and out of spec

y = norm.pdf(x, 0, 1)
y2 = norm.pdf(x_all, 0, 1)

fig, ax = plt.subplots( figsize=(9, 6) )
plt.style.use('fivethirtyeight')
ax.plot(x_all, y2)

ax.fill_between(x, y, 0, alpha=0.3, color='b')
ax.fill_between(x_all, y2, 0, alpha=0.1)
ax.set_xlim([-2, 2])
ax.set_xlabel('# of Standard Deviations Outside the Mean')
ax.set_yticklabels([])
ax.set_title('Normal Gaussian Curve')

plt.savefig('normal_curve.png', dpi=72, bbox_inches='tight')
plt.show()
'''
