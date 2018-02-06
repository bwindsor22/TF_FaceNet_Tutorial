#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:56:35 2017
"""

save_dir = '/medium-facenet-tutorial/temp_model/'

#%%

import tensorflow as tf

tf.reset_default_graph()

with tf.Session() as sess:
    #Prepare to feed input, i.e. feed_dict and placeholders
    w1 = tf.placeholder("float", name="w1")
    w2 = tf.placeholder("float", name="w2")
    b1= tf.Variable(2.0,name="bias")
    feed_dict ={w1:4,w2:8}
    
    #Define a test operation that we will restore
    w3 = tf.add(w1,w2)
    w4 = tf.multiply(w3,b1,name="op_to_restore")
    
    sess.run(tf.global_variables_initializer())
    
    print(sess.run(w4,feed_dict))
    
    #Create a saver object which will save all the variables
    
    saver = tf.train.Saver()
    
    
    save_path = saver.save(sess, save_dir + "model.ckpt")
    print("Model saved in file: %s" % save_path)

#%%
with tf.Session() as sess:
    # Restore variables from disk.
    saver = tf.train.import_meta_graph( save_dir + 'model.ckpt.meta' )
    saver.restore(sess, save_dir + "model.ckpt")
    print("b1 : %s" % b1.eval())

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    feed_dict ={w1:13.0,w2:17.0}
     
    #Now, access the op that you want to run. 
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
     
    print(sess.run(op_to_restore,feed_dict))
