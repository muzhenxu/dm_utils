import sys
sys.path.append('../hold_out/')
import hold_out as ho
import pandas as pd


df  = pd.DataFrame({"name":["lh",'fd','cjy'],"age":[11,12,13]})

train,test=ho.hold_out(df, train_size=0.8, suffle=True, random_state = 2017)
print ("ok")

print (train)
print ("test")
print (test)
