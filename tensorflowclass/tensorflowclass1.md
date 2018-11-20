#class1 fundamentals
1.初级构造：计算图（computation graph） 箭头代表数据流，节点代表计算（加、减、乘、除等）
数据流可以被重复使用。  
2.fundamental tensorflow workflow：  

```python
import tensorflow as tf
/#tf.placeholder ,we must give it value when we run our model
a = tf.placeholder(tf.int32,name="input")
b = tf.placeholder(tf.int32,name="input")

c = tf.add(a,b,name="add")
d = tf.multiply(a,b,name="multiply")
out = tf.add(c,d,name="output")

sess = tf.Session()
feed_dict = {a:4,b:3}
/#sess.run接受两个参数：’fetches‘列表存放输出节点，’feed_dict‘填充tf.placeholder
result = sess.run(out,feed_dict=feed_dict)
print("({0}*{1}) + ({0}+{1}) = {2}".format(feed_dict[a], feed_dict[b], result))
sess.close()
```   

3.Tensorflow Core API  

* tf.Tensor
* tf.Operation
* tf.Graph
* tf.Session
* tf.Variable  

3.1 什么是张量（tensor）？  
张量即n维矩阵，0维张量为标量（scalar），1维张量为向量（vector），2为张量为矩阵。通过Tensorflow传递的每一个值都是tensor对象-即张量的tensorflow表现形式。  
3.1.2 如何定义张量？  
有两种方式定义张量：Naive Python types和Numpy arrays（recommended）这两种都能自动转换为tensor对象。  
3.1.3 Tensors from Naive Python  
tf.constant creates a tf.Tensor from a fixed value  

	tf.constant(array)
3.1.4 Tensors from Numpy array

	tf.constant(np.array)  
3.1.5 np.array and np.asarray is recommended.原因是numpy矩阵可以定义数据类型，而且Tensorflow与Numpy紧密集成。  

```python
(tf.float32 == np.float32 and
 tf.float64 == np.float64 and
 tf.int8 == np.int8 and
 tf.int16 == np.int16 and
 tf.int32 == np.int32 and
 tf.int64 == np.int64 and
 tf.uint8 == np.uint8 and
 tf.bool == np.bool and
 tf.complex64 == np.complex64)
```  

**注意：当创建字符串张量时必须使用标准的python lists**

>tf_string_tensor = [b"first", b"second", b"third"]



tf的tensor对象显示的信息只有常量名，shape和数据类型，如果要显示具体的tensor对象的值，需要以下语句：
>tf.Session().run(tensor_object)  

3.2 tensorflow Operation Objeccts(tf Ops)  
计算节点 当Ops创建后并不执行，必须使用tf.Session().run()方法执行。  
3.3 tensorflow Graph Objects  
当import tensorflow时，它会自动创建一个默认的Graph对象。
我们可以创建新的Graph通过以下方法：

```python
new_graph = tf.Graph()

with new_graph.as_default():
    a = tf.add(3,4)
    b = tf.multiply(a,b)

sess = tf.Session(graph=new_graph)
sess.run(b)
sess.close()#clean up the session	
```  

3.4 tensorflow Session  
3.4.1 create Sessions  

```python
sess_default = tf.Session()#using the default graph equivalent to:Session(graph=tf.get_default_graph())

new_graph = tf.Graph()
sess_new = tf.Session(graph=new_graph)

```  
3.4.2 running Sessions  

```python
sess_default.run(fetches,feed_dict)#fetches:a list of Tensors/Ops
sess_default.close()#close the Sessions
sess_new.close()
```  

3.5 tensorflow Variable Objects  

```python
my_var = tf.Variable(0,name='my_var')#create a Variable
increment = tf.assign(my_var,my_var+1)#do increment on the value

init = tf.global_variables_initializar()#Variable initial Ops. Important！

sess=tf.Session()
sess.run(init)#initialize the Variable Important!
for i in range(10):
    print(sess.run(increment),end=" ")
sess.close()

```  

3.6 using tensorboard
  
```python
a_summ = tf.summary.scalar("a_input_summary",a)#add summaries
merged = tf.summary.merge_all()#group all summaries together
tf.summary.FileWriter("Path",graph=graph)#output the graph
```  

```bash
tensorboard --logdir=Path
```  
3.7 Organizing graphs with tf.name_scoope()  

```python
with tf.name_scope('name1'):
    #ops
```  

3.8 A complete tensorflow workthrough  

```python
#define a script to keep track of different runs of our graphs,it will create dir like this:
#tbout/ex1/1 first run
#tbout/ex1/2 second run
import os.path as osp
import itertools

def get_fresh_dir(parent_dir):
    ''' get an unused directory name in parent_dir '''
    import itertools as it
    import os.path as osp
    for i in it.count():
        possible_dir = osp.join(parent_dir, str(i))
        if not osp.exists(possible_dir):
            return possible_dir
#buid the model in a Graph
ex1_graph = tf.Graph()
with ex1_graph.as_default():
    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.int32,name="input_1")
        y = tf.placeholder(tf.int32,name="input_2")
        z = tf.placeholder(tf.int32,name="input_3")
    with tf.name_scope("inputs_summaries"):
        x_summ = tf.summary.scalar("x_input_summary",x)
        y_summ = tf.summary.scalar("y_input_summary",y)
        z_summ = tf.summary.scalar("z_input_summary",z)
    with tf.name_scope("multiplication_product"):
        a = tf.multiply(x,y,name="mul")
        b = tf.reduce_prod([x,y,z],name='product')
        c = tf.multiply(y,z,name="mul")
    with tf.name_scope("mul_prod_summaries"):
        a_summ = tf.summary.scalar("a_summary",a)
        b_summ = tf.summary.scalar("b_summary",b)
        c_summ = tf.summary.scalar("c_summary",c)
    with tf.name_scope("sum"):
        d = tf.reduce_sum([a,b,c],name="sum")
    with tf.name_scope("sum_summaries"):
        d_summ = tf.summary.scalar("sum_summary",d)
        
    with tf.name_scope("sum_holder"):#a Varaible to keep track of sum
        sum_holder = tf.Variable(0, trainable=False, dtype=tf.int32, name="sum_holder")
        inc_sum = tf.assign(sum_holder,sum_holder+d,name="increment_sum")
    with tf.name_scope('global_step'):#global step Variable to keep track of number of runs
        global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_step")
        inc_step = tf.assign(global_step, global_step + 1, name="increment_step")
    with tf.name_scope('helpers'):
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

#run the model and gather summaries
sess = tf.Session(graph=ex1_graph)
writer = tf.summary.FileWriter(get_fresh_dir('tbout/ex1'), 
                               graph=ex1_graph)
sess.run(init)
for i in range(5):
    rand_x = random.uniform(0,20)
    rand_y = random.uniform(0,20)
    rand_z = random.uniform(0,20)
    feed_dict = {x:rand_x,y:rand_y,z:rand_z}
    step,sum_holders,summaries = sess.run([inc_step,inc_sum,merged],feed_dict)
    print(sum_holders)
    writer.add_summary(summaries, global_step=step)#gather summaries
    writer.flush()
print(sum_holders)
print(summaries)
writer.close()
sess.close()
```

































