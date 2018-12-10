#tensorflowclass3
1.learning an and-gate

```python
class TF_Logistic_gate():
    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope('input'):
                self.x1 = tf.placeholder(tf.float32,name='x1')
                self.x2 = tf.placeholder(tf.float32,name='x2')
                self.label = tf.placeholder(tf.float32,name='label')
                self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
            with tf.name_scope('model'):
                self.w1 = tf.Variable(tf.random_normal([]),name='w1')
                self.w2 = tf.Variable(tf.random_normal([]),name='w2')
                self.b = tf.Variable(0.0,dtype=tf.float32,name='b')
                self.output = tf.nn.sigmoid(self.w1*self.x1+self.w2*self.x2+self.b,name='output')
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(self.label-self.output),name='loss')
                self.correct = tf.equal(tf.round(self.output),self.label)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct,tf.float32))
            with tf.name_scope('train'):
                self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=graph)
        self.sess.run(self.init)
    
    def fit(self,train_dict):
        return self.sess.run([self.loss,self.accuracy,self.train],feed_dict=train_dict)
    def predict(self,test_dict):
        return self.sess.run([self.w1,self.w2,self.b,self.output],feed_dict=test_dict)
 
            
my_and_gate = TF_Logistic_gate()#创建逻辑门对象

my_and_table = np.array([[0,0,0],
                     [1,0,0],
                     [0,1,0],
                     [1,1,1]])
#指明obj.x1非常重要
my_train_dict={my_and_gate.x1: and_table[:,0],
            my_and_gate.x2: and_table[:,1],
            my_and_gate.label: and_table[:,2], 
            my_and_gate.learning_rate: 0.5}

my_test_dict = {my_and_gate.x1: and_table[:,0], #[0.0, 1.0, 0.0, 1.0], 
                 my_and_gate.x2: and_table[:,1]} # [0.0, 0.0, 1.0, 1.0]}
snapshots = []

for epoch in range(5000):
    loss,acc,_ = my_and_gate.fit(my_train_dict)
    if epoch%1000==0 or epoch<10:
        print(f"{epoch}: loss{loss} acc{acc}")
        w1,w2,b,output = my_and_gate.predict(my_test_dict)
        snapshots.append([w1,w2,b,np.round(output)])
print(f"{epoch}: loss{loss} acc{acc}")      
print(snapshots[-1])
```

2.create your own neural network

```python

nn2_graph = tf.Graph()
with nn2_graph.as_default():
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32,[None,200],name='x')
        y = tf.placeholder(tf.float32,[None,100],name='y')
    with tf.name_scope('hidden1'):
        w1 = tf.Variable(tf.truncated_normal([200,800]),name='W1')
        b1 = tf.Variable(tf.zeros([800]),name='b')
        a1 = tf.matmul(x,w1)+b1
    with tf.name_scope('hidden2'):
        w2 = tf.Variable(tf.truncated_normal([800,600]),name='W2')
        b2 = tf.Variable(tf.zeros([600]))
        a2 = tf.matmul(a1,w2)+b2
    with tf.name_scope('hidden3'):
        w3 = tf.Variable(tf.truncated_normal([600,400]),name='W3')
        b3 = tf.Variable(tf.zeros([400]),name='b3')
        a3 = tf.matmul(a2,w3)+b3
    with tf.name_scope('output'):
        w4 = tf.Variable(tf.truncated_normal([400,100]),name='W4')
        b4 = tf.Variable(tf.zeros([100]),name='b4')
        output = tf.matmul(a3,w4)+b4
    with tf.name_scope("global_step"):
        global_step = tf.Variable(0.0,trainable=False,name = 'global_step')
        inc_step = tf.assign(global_step,global_step+1,name = 'inc_step')
    with tf.name_scope('summaries'):
        for var in tf.trainable_variables():
            hist_summary=tf.summary.histogram(var.op.name,var)
        summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

sess = tf.Session(graph=nn2_graph)
writer = tf.summary.FileWriter(helpers_03.get_fresh_dir(tb_base_path),graph=nn2_graph)
sess.run(init)
summaries = sess.run(summary_op)
writer.add_summary(summaries)
writer.close()
sess.close()
```