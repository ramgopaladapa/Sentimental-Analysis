##project by pazim goyal
print('Title: Sentiment Analysis... On Product Reviews ')
print('\nStage 1 : Dataset Loading\n')

#pandas csv reading library
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
import time
import numpy as np

from math import sqrt,log

start=time.process_time()
#testing and training data files 
file_name="training.csv"
file_name2="testing.csv"
file_name3="testing_small.csv"
#------------------------------preprocessing-------------------------
#testing data
print("\nLoading Review Data from CSV file.......")
abc=pandas.read_csv(file_name,names=['rate','txt'])
print("Data Loaded")

print('\nStage 2 : Preporcessing')


#stop words removal
print("\nLoading Stop words ")
files = open('stopwords.txt', 'r') 
setz= files.read()
print("Stop Words Loaded")
review=list()
reviewx=list()
print("Removing Stop words From Review DataSet.....")
for i in abc.txt:
    i=i.replace(","," ").replace("."," ").replace(";"," ").replace("#"," ").replace(")"," ").replace("("," ")
    review.append([w for w in  i.lower().split(" ") if w not in setz]) #taking a review and pazim  splitting it to words and removing all the stop words
print("Stop words Removed \nCounting total positive and Negative Classes")

#for every in review: new dataframe with stop words removed        
df=DataFrame({'text':review,'gb':abc.rate.values})
abcd=pandas.read_csv(file_name2,names=['text','real'])
review2=list()
print("\nRemoving Stop words From Testing DataSet.....")
for i in abcd.text:
    i=i.replace(","," ").replace("."," ").replace(";"," ").replace("#"," ").replace(")"," ")
    review2.append([w for w in  i.lower().split(" ") if w not in setz]) #taking a review and pazim  splitting it to words and removing all the stop words


print("Time taken For Preprocessing {}".format(time.process_time()-start))

