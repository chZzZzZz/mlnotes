#tensorflow class2
##uci wine quality prediction
###part1:access the data

```python
import os.path as osp
import os
import requests

def urlretrieve(url, path):
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)
    return path, r.headers

def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)

def download(url, _dir):
    if not osp.exists(_dir):
        mkdir(_dir)

    filename = url.rsplit('/',1)[1]
    fullpath = osp.join(_dir, filename)
    if not osp.exists(fullpath):
        urlretrieve(url, fullpath)
    return fullpath

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
data_dir="data"
download(url,data_dir)
```  

###part2:read csv

```python
import pandas as pd
data = pd.read_csv('data/winequality-white.csv',sep=':',dtype='float')
```  
###part3:create train_test_split

```python
def df_train_test_blocks(df,pct):
    num_train_samples = int(len(df)*pct)
    return df[:num_train_samples],df[num_train_samples:]

import random
features = data.columns[:-1]
subset = random.sample(list(np.arange(len(features)-1),5)
fea_sub = features[subset]
my_data = data[list[fea_sub]+'quality']
train,test=df_train_test_blocks(my_data,0.8)
x_train = train[fea_sub]
y_train = train['quality']
x_test = test[fea_sub]
y_test = test['quality']
```  
###part4:create model

```python
import tensorflow as tf
class TF_MV_linearRegression()
    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope('input'):
                self.x = tf.placeholder(tf.float32,[None,5],name='x')#x任意行5列
                self.y = tf.placeholder(tf.float32,[None,1],name='y')
                self.learning_rate = tf.placeholder(tf.float32,[],name='learning_rate')
            with tf.name_scope('model'):
                self.w = tf.Variable(tf.truncated_normal([5,1]),name='w')#从截断的正态分布中选取随机值
                self.b = tf.Variable(0.0,name='b')
                self.y_hat = tf.matmul(w,self.x)+self.b
                
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(self.y,self.y_hat),name='MSE')
                
            with tf.name_scope('train'):
                self.train = tf.train.GradientDecentOptimizer(self.learning_rate).minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.sess = tf.Session(graph=graph)
            self.sess.run(init)
    def fit(self,train_dict):
        return sess.run([self.loss,self.w,self.b,self.train],feed_dict=train_dict)
    def predictt(self,test_dict):
        return sess.run(self.y_hat,feed_dict=test_dict)
```  

###part5:train and test model

```python
my_model = TF_MV_linearRegression()
my_train_dict = {my_model.x:x_train.values,
                 my_model.y:y_train.values,
                 my_model.learning_rate:0.01}
my_test_dict = {my_model.x:x_test.values}

snapshots = []
for epoch in range(1000):
    loss,w,b,_ = my_model.fit(train_dcit)
    if epoch%100 == 0 or epoch<10:
        print(f"{epoch} {loss}")
        y_test = my_modle.predict(test_dict)
        snapshots.append([epoch,y_test,w,b,loss])
print(f"{epoch} {loss}")
```