import sys
sys.path.append('../hold_out/')
import hold_out as ho
import pandas as pd


df  = pd.DataFrame({"name":["lh",'fd','cjy'],"age":[11,12,13]})

train,test=ho.hold_out(df, train_size=0.8, suffle=True, random_state = 2017)
print ("train")
print (train)
print ("test")
print (test)


#test case two

df2 = pd.read_csv("./resource/mdf4_for_tmp.csv")

train2,test2=ho.hold_out(df2, train_size=0.8, suffle=False, random_state = 2019)
print (df2.head())
print ("train2")
print (len(train2))
print ("test2")
print (len(test2))
print (train2.head())
