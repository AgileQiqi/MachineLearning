#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


import tensorflow as tf


# In[3]:


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#one_hot is used to encode lable, the format requirement of the logit layer of NN


# In[5]:


#Define hyperparameter
learning_rate=0.5
epochs=10
batch_size=100

#Placeholder
#Input images with 28*28 digit, Pixel = 784
#x = tf.placeholder(tf.float32, [None,784])
x = tf.placeholder(tf.float32)
#Output as 0-9 one-hot encode 
#y = tf.placeholder(tf.float32, [None,10])
y = tf.placeholder(tf.float32)


# In[6]:


#Define parameter w&b

#Parameter of hidden layer w(weight),b(bias)
w1=tf.Variable(tf.random_normal([784,300],stddev=0.03), name='w1')
b1=tf.Variable(tf.random_normal([300]), name='b1')

#Parameter of Output layer
w2=tf.Variable(tf.random_normal([300,10],stddev=0.03),name='w2')
b2=tf.Variable(tf.random_normal([10]), name='b2')

#w & b are intialized randomly, tf.random_normal() generate number of normal distribution


# In[7]:


#Build the hidden layer
hidden_out=tf.add(tf.matmul(x,w1),b1)
#Using relu as the activation function
hidden_out=tf.nn.relu(hidden_out)

#The above formula is:
#z=wx+b
#h=relu(z)


# In[8]:


#Calculate the output
y_=tf.nn.softmax(tf.add(tf.matmul(hidden_out,w2), b2))
#softmax is the activation function for output, mostly used in the Multiclass Classification(single label)


# In[ ]:


#Back Propagation(BP)


# In[9]:


#Loss- Cross Entropy
y_clipped=tf.clip_by_value(y_,1e-10,0.9999999)
#tf.clip_by_value() used to clip the value of y_, ensure the range y_ from min to max, 
#value < min will be set as min, value >max will be set as max
#max is 0.9999999 because of the size of float32

#1. Calculate the cross entropy for n labels
#2. Calculate the mean of m samples
cross_entropy= -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_clipped)+(1-y)*tf.log(1-y_clipped), axis=1))


# In[10]:


#Define optimize algorithm
#optimizer has multiple choices
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


# In[11]:


#Initialize the operator
init_op=tf.global_variables_initializer()


# In[12]:


#Create the accuracy node
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#Mean of accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#tf.cast used for type cast


# In[28]:


#Setting up the training


# In[14]:





# In[15]:


#Create session
with tf.Session() as sess:
    #Init variables
    sess.run(init_op)
    total_batch=int(len(mnist.train.labels)/batch_size)
    for epoch in range(epochs):
        avg_cost=0
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
            _, c=sess.run([optimiser,cross_entropy],feed_dict={x:batch_x,y:batch_y})
            avg_cost += c/total_batch
        print("Epoch:", (epoch+1),"cost = ", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels}))


# In[ ]:




