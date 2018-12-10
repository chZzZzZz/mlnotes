#tensorflow class4
1.定义函数来创建神经网络层

```python
def fully_connected_layer(incoming_layer,#上一层
                         num_nodes,#这一层的节点数
                         activation_fn=tf.nn.sigmoid,#激励函数
                         w_stddev=0.5,#初始化w的正态分布的标准差
                         b_val=0.0,#初始化b
                         keep_prob=None,#此节点的保留概率
                         name=None)
    incoming_layer = tf.convert_to_tensor(incoming_layer)
    prev_num_nodes = incoming_layer.shape.dims[-1].value

    with tf.name_scope("fully_connected_layer"):
        W =       tf.Variables(tf.truncated_normal([prev_num_nodes,num_nodes],stddev=w_stddev),name='W')
        b = tf.Variables(tf.constant(b_val,shape=[num_nodes]),name='b')
        z = tf.matmul(incoming_layer,W)+b

    a = activation_fn(z) if activation_fn is not None else z 
    final_a = a if keep_prob is None else tf.nn.dropout(a,keep_prob)
    return final_a
``` 

2.Flattening Layers
The below code is another commonly used type of layer. It "flattens" a tensor (excluding the first dimension, which is the number of batches). For example, a matrix of ten 28x28 RGB images normally has a shape like this: [10, 28, 28, 3]. However, our model expects inputs to only be dimension 2. By flattening it, we "string out" the input pixels into a vector of length 28*28*3=2352, so our final output shape is [10, 2352]  

```python
def flatten(incoming,name=None):
    flat_shape = [-1,np.prod(incoming[1:].value)]
    return tf.reshape(incoming,flat_shape)
```  

3.Exercise on MNIST Dataset  

train_data.shape-(60000,28,28,1)  test_data.shape-(10000,28,28,1)

##define the model

```python
class MNIST_Model(object):
    def __init__(s):
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("input"):
                s.x = tf.placeholder(tf.uint8,shape=[None,28,28,1],name='x')
                s.y = tf.placeholder(tf.int32,shape=[None])
            with tf.name_scope("hyperparams"):
                s.learning_rate = tf.placeholder(tf.float32,[],name='learning_rate')
                s.momentum = tf.placeholder(tf.float32,[],name='momentum')
                s.x_keep_prob = tf.placeholder(tf.float32,[],name='x_keep_prob')
                s.h_keep_prob = tf.placeholder(tf.float32,[],name='h_keep_prob')
            with tf.name_scope("preprocess"):
                x_flat = flatten(s.x)
                x_float = tf.cast(x_flat,tf.float32)
                s.x_dropped = tf.nn.dropout(x_float,x_keep_prob)
                s.one_hot_labels = tf.one_hot(s.y,10)
            with tf.name_scope("model"):
                make_fc = fully_connected_layer#abbreviation
                s.h1 = make_fc(s.x_dropped,1200,activation_fn=tf.nn.relu, w_stddev=0.001, 
                          b_val=0.1,keep_prob=s.h_keep_prob, name='h1')
                s.h2 = make_fc(s.h1,1200, activation_fn=tf.nn.relu,w_stddev=0.001, 
                          b_val=0.1,keep_prob=s.h_keep_prob, name='h2')
                s.h3 = make_fc(s.h2,1200, activation_fn=tf.nn.relu, w_stddev=0.001, 
                          b_val=0.1,keep_prob=s.h_keep_prob, name='h3')
                s.out = make_fc(s.h3, 10, activation_fn=None,w_stddev=0.001, 
                          b_val=0.1, name='out')
            with tf.name_scope("loss"):
                smce = tf.nn.softmax_cross_entropy_with_logits_v2
                s.loss = tf.reduce_mean(smce(logits=s.out,labels=s.one_hot_labels))
            with tf.name_scope("train"):
                opt = tf.trian.MomentumOptimizer(s.learning_rate,s.momentum)
                s.train = opt.minimize(s.loss)
            with tf.name_scope("global_step"):
                global_step = tf.Variable(0,trainable=False,name='global_step')
                s.inc_step = tf.assign_add(global_step,1,name='inc_step')
            with tf.name_scope("prediction"):
                s.softmax = tf.nn.softmax(s.out,name='softmax')
                s.prediction = tf.cast(tf.argmax(s.softmax,1),tf.int32)
                s.pred_correct = tf.equal(s.y,s.prediction)
                s.pred_accuracy = tf.reduce_mean(tf.cast(s.pred_correct,tf.float32))
            s.init = tf.global_variables_initializar()

        s.sess = tf.Session()
        s.sess.run(s.init)

    def fit(s,trian_dict):
        tr_loss,step,tr_acc,_ = s.sess.run([s.loss,s.inc_step,s.pred_accuracy,s.train],feed_dict=train_dict)

    def predict(s,test_dict):
        ct_correct,preds = s.sess.run([s.pred_correct,s.prediction],feed_dict=test_dict)

```

##train the model

```python
mm = MNIST_Model()
for epoch in range(15):
    for batch_data,batch_labels in batches(train_data,train_labels,100):
        m = min(.5+epoch*0.001,0.99)
        train_dict = {mm.x:batch_data,
                      mm.y:batch_labels,
                      mm.learning_rate:0.01,
                      mm.momentum:m,
                      mm.x_keep_prob:0.8,
                      mm.h_keep_prob:0.5}
        tr_loss,step,tr_acc = mm.fit(train_dict)
    info_update = "Epoch:{} Step:{} Loss:{} Acc:{}"
    print(info_update.format(epoch,step,tr_loss,tr_acc))
```

##test the model
```python
batch_correct_cts = []
for batch_data,batch_labels in batches(test_data,test_labels,200):
    test_dict = {mm.x : batch_data,   mm.x_keep_prob : 1.0,
                 mm.y : batch_labels, mm.h_keep_prob : 1.0}#here keep_prob must be one
    correctness, curr_preds = mm.predict(test_dict)
    batch_correct_cts.append(correctness.sum())

print(sum(batch_correct_cts)/len(test_data))
```
