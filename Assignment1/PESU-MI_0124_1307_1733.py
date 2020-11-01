'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

import math

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large


def get_entropy_of_dataset(df):
  p = 0
  n = 0
  entropy = 0
  for i in df[df.columns[len(df.columns)-1]]:
    if i == 'yes':
      p = p+1
    elif i=='no':
      n = n+1
  p1 = p/(p+n)
  n1 = n/(p+n)
  #entropy = (-p1 * np.log2(p1)) - (n1 * np.log2(n1))         #Idk why it's giving a different answer. -Compared to math.log
  entropy = (-p1 * math.log(p1,2)) - (n1 * math.log(n1,2))
  return entropy




'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
	entropy_of_attribute = 0
	entropy = {}
	for index in range(len(df[attribute])):
		if df[attribute][index] in entropy.keys():
			if df[df.columns[len(df.columns)-1]][index] == 'yes':
				entropy[df[attribute][index]][0] += 1                       # no of positives ++
			else:
				entropy[df[attribute][index]][1] += 1                       # no of negatives ++
			entropy[df[attribute][index]][2] += 1                           # increment total count
		else:
			if df[df.columns[len(df.columns)-1]][index] == 'yes':
				entropy[df[attribute][index]] = [1, 0, 1]                       # no of positives ++
			else:
				entropy[df[attribute][index]] = [0, 1, 1]                      # no of negatives ++

	for k, v in entropy.items():
		ent = 0
		pi = v[0]/ v[2]
		ni = v[1]/ v[2]
		if ni > 0 and pi>0:
			ent = (-pi * math.log(pi,2)) - (ni * math.log(ni,2))
		entropy_of_attribute += ( (v[0] + v[1])/len(df))  * ent
	return abs(entropy_of_attribute)
#get_entropy_of_attribute(df, 'outlook')



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	information_gain = 0

	en = get_entropy_of_dataset(df)
	info = get_entropy_of_attribute(df,attribute)

	information_gain = en - info
	return information_gain
#get_information_gain(df,'humidity')

'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_selected_attribute(df):

    information_gains={}
    selected_column=''
    count = 0

    for (columnName, columnData) in df.iteritems():
        count+=1
        if(columnName!='play'):
            information_gains[columnName] = get_information_gain(df,columnName)

    #print(information_gains)
    high = -100
    for i in information_gains:
        if information_gains[i] > high:
            high = information_gains[i]
            selected_column = i

    return (information_gains,selected_column)





'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
