
from preprocessing import *
import numpy as np

start=time.process_time()
print('\nStage 3 : Algorithm Implementaion')

print('\n Using S.V.M \n')
poss = dict()
negs=dict()
print("Counting Vocabulary for SVM")
poss.clear()
negs.clear()
for row_index,row in df.iterrows():
    if row['gb']==1:
        for i in row['text']:
            if i not in poss:
                poss[i]=1
                negs[i]=0
            else:   
                poss[i]+=1
    else:
        for i in row['text']:
            if i not in negs:
                poss[i]=0
                negs[i]=1
            else:   
                negs[i]+=1
print("Total Vocabulary Count  : {} ".format(len(poss)))

#---------------------------------testing-----------------------------

result=[]
result.clear()
poscount=0
negcount=0
revcount=0
for s in review2:
    count1=0
    count2=0
    for v in s:
        if poss.get(v,0)>negs.get(v,0):
            count1=count1+1
        elif poss.get(v,0)<negs.get(v,0):
            count2=count2+1        
    if count1>count2:
        result.append(1)
        revcount=revcount+1
        poscount=poscount+1
    else:
        result.append(0)
        revcount = revcount + 1
        negcount = negcount + 1

dfpos=DataFrame({'real':abcd.real.values,'result':result,'review':abcd.text})
dfpos.to_csv('predictions2.csv')

true=0
false=0
for row_index,row in dfpos.iterrows():
    if row['real'] == row['result']:
        true =true+1
    else:
        false=false+1
print('\nStage 4 : Results')

svmacc = (true*100)/(true+false)
print("Accuracy for SVM : {} %".format((true*100)/(true+false)))
print("Time taken for SVM : {} seconds".format(time.process_time()-start))        
print("Results stored in prediction2.csv")

labels = 'Accuracy','error'
sizes = [true,false]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()


#barchart

objects = ('Total Review', 'Positive Reviews', 'Negative Reviews')
y_pos = np.arange(len(objects))
performance = [revcount,poscount,negcount]

plt.bar(y_pos, performance, align='center', alpha=0.5,color=['blue', 'green','red'])
plt.xticks(y_pos, objects)
plt.ylabel('Reviews Count')
plt.title('Sentiment Analysis using SVM')

plt.show()




#Naivebayes



print("\nNaive Bayes\n")
start = time.process_time()
# count +ve and -ve reviews
positive = 0
negative = 0
for i in abc.rate:
    if i == 1:
        positive = positive + 1
    else:
        negative = negative + 1
total = positive + negative
pofp = positive / total  # Priors of positive
pofn = negative / total  # priors of negative
print("Total Positive Reviews : {}\nTotal Negative Reviews : {}\nP(positive) : {}\nP(negative) {}: ".format(positive,
                                                                                                            negative,
                                                                                                            pofp, pofn))

# --------------training of algo----------------------------------------------
temp2 = dict()
temp = dict()
print("Counting Vocabulary for positive and negative reviews")

for row_index, row in df.iterrows():
    tem = row['gb']
    if tem == 1:
        for i in row['text']:
            if i not in temp:
                temp[i] = 1
            else:
                temp[i] = temp[i] + 1


    elif tem == 0:
        for i in row['text']:
            if i not in temp2:
                temp2[i] = 1
            else:
                temp2[i] = temp2[i] + 1

print("Vocab Size of positive words {} ".format(len(temp)))
print("Vocab Size of negative words {} ".format(len(temp2)))
vocab = list()
vocab = temp.copy()
vocab.update(temp2)
print("Total Vocabulary Size : {} ".format(len(vocab)))

ppos = dict()
pneg = dict()
summ = 0
sumn = 0
for i in temp:
    summ = summ + temp[i]
for i in temp2:
    sumn = sumn + temp2[i]
print("\nTotal Words in Positive reviews : {}\nTotal Words in Negative reviews : {}".format(summ, sumn))

# -----------------------------Testing-------------------------------------------------
lastlist = list()

for each in review2:
    ptemp = pofp
    p2temp = pofn
    for i in each:
        ptemp = ptemp * (temp.get(i, 0) + 1) / (summ + len(vocab))
    for i in each:
        p2temp = p2temp * (temp2.get(i, 0) + 1) / (sumn + len(vocab))
    if ptemp > p2temp:
        lastlist.append(1)
        revcount = revcount + 1
        poscount = poscount + 1
    else:
        lastlist.append(0)
        revcount = revcount + 1
        negcount = negcount + 1

dfpos = DataFrame({'real': abcd.real.values, 'count': lastlist, 'rev': abcd.text})
dfpos.to_csv('predictions1.csv')

true = 0
false = 0
for row_index, row in dfpos.iterrows():
    if row['real'] == row['count']:
        true = true + 1
    else:
        false = false + 1

nbacc = (true * 100) / (true + false)
print("Accuracy for Naive Bayes : {} %".format((true * 100) / (true + false)))
print("Time taken for naive bayes : {} seconds".format(time.process_time() - start))
print("Results stored in prediction1.csv")

labels = 'Accuracy', 'error'
sizes = [true, false]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()



#barchart

objects = ('Total Review', 'Positive Reviews', 'Negative Reviews')
y_pos = np.arange(len(objects))
performance = [revcount,poscount,negcount]

plt.bar(y_pos, performance, align='center', alpha=0.5,color=['blue', 'green','red'])
plt.xticks(y_pos, objects)
plt.ylabel('Reviews Count')
plt.title('Sentiment Analysis using Naivebayes')

plt.show()


#barchart

objects = ('SVM', 'Naive Bayes')
y_pos = np.arange(len(objects))
performance = [svmacc,nbacc]

plt.bar(y_pos, performance, align='center',color=['green', 'blue'])
plt.xticks(y_pos, objects)

plt.ylabel('Percentage')
plt.title('Sentiment Analysis Accuracy Copmarison using SVM,NaiveBayes')

plt.show()