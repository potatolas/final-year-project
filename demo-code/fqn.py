import tensorflow as tf 

def printFQN(): 
    try:
        if(type(tf.Variable).__name__ == "function"): 
            function_name = tf.Variable.__name__
            print(tf.Variable.__module__ + ": " + function_name + "()")
        else: 
            print(tf.Variable.__module__ + "." + tf.Variable.__name__ + ": <init>")
    except:
        print("tf.Variable") 
    try:
        if(type(tf.truncated_normal).__name__ == "function"): 
            function_name = tf.truncated_normal.__name__
            print(tf.truncated_normal.__module__ + ": " + function_name + "()")
        else: 
            print(tf.truncated_normal.__module__ + "." + tf.truncated_normal.__name__ + ": <init>")
    except:
        print("tf.truncated_normal") 
    try:
        if(type(tf.Variable).__name__ == "function"): 
            function_name = tf.Variable.__name__
            print(tf.Variable.__module__ + ": " + function_name + "()")
        else: 
            print(tf.Variable.__module__ + "." + tf.Variable.__name__ + ": <init>")
    except:
        print("tf.Variable") 
    try:
        if(type(tf.constant).__name__ == "function"): 
            function_name = tf.constant.__name__
            print(tf.constant.__module__ + ": " + function_name + "()")
        else: 
            print(tf.constant.__module__ + "." + tf.constant.__name__ + ": <init>")
    except:
        print("tf.constant") 
    try:
        if(type(tf.nn.conv2d).__name__ == "function"): 
            function_name = tf.nn.conv2d.__name__
            print(tf.nn.conv2d.__module__ + ": " + function_name + "()")
        else: 
            print(tf.nn.conv2d.__module__ + "." + tf.nn.conv2d.__name__ + ": <init>")
    except:
        print("tf.nn.conv2d") 
    try:
        if(type(tf.nn.max_pool).__name__ == "function"): 
            function_name = tf.nn.max_pool.__name__
            print(tf.nn.max_pool.__module__ + ": " + function_name + "()")
        else: 
            print(tf.nn.max_pool.__module__ + "." + tf.nn.max_pool.__name__ + ": <init>")
    except:
        print("tf.nn.max_pool") 
    try:
        if(type(tf.nn.relu).__name__ == "function"): 
            function_name = tf.nn.relu.__name__
            print(tf.nn.relu.__module__ + ": " + function_name + "()")
        else: 
            print(tf.nn.relu.__module__ + "." + tf.nn.relu.__name__ + ": <init>")
    except:
        print("tf.nn.relu") 
    try:
        if(type(tf.reshape).__name__ == "function"): 
            function_name = tf.reshape.__name__
            print(tf.reshape.__module__ + ": " + function_name + "()")
        else: 
            print(tf.reshape.__module__ + "." + tf.reshape.__name__ + ": <init>")
    except:
        print("tf.reshape") 
    try:
        if(type(tf.matmul).__name__ == "function"): 
            function_name = tf.matmul.__name__
            print(tf.matmul.__module__ + ": " + function_name + "()")
        else: 
            print(tf.matmul.__module__ + "." + tf.matmul.__name__ + ": <init>")
    except:
        print("tf.matmul") 
    try:
        if(type(tf.nn.relu).__name__ == "function"): 
            function_name = tf.nn.relu.__name__
            print(tf.nn.relu.__module__ + ": " + function_name + "()")
        else: 
            print(tf.nn.relu.__module__ + "." + tf.nn.relu.__name__ + ": <init>")
    except:
        print("tf.nn.relu") 
    try:
        if(type(tf.placeholder).__name__ == "function"): 
            function_name = tf.placeholder.__name__
            print(tf.placeholder.__module__ + ": " + function_name + "()")
        else: 
            print(tf.placeholder.__module__ + "." + tf.placeholder.__name__ + ": <init>")
    except:
        print("tf.placeholder") 
    try:
        if(type(tf.reshape).__name__ == "function"): 
            function_name = tf.reshape.__name__
            print(tf.reshape.__module__ + ": " + function_name + "()")
        else: 
            print(tf.reshape.__module__ + "." + tf.reshape.__name__ + ": <init>")
    except:
        print("tf.reshape") 
    try:
        if(type(tf.placeholder).__name__ == "function"): 
            function_name = tf.placeholder.__name__
            print(tf.placeholder.__module__ + ": " + function_name + "()")
        else: 
            print(tf.placeholder.__module__ + "." + tf.placeholder.__name__ + ": <init>")
    except:
        print("tf.placeholder") 
    try:
        if(type(tf.float32).__name__ == "function"): 
            function_name = tf.float32.__name__
            print(tf.float32.__module__ + ": " + function_name + "()")
        else: 
            print(tf.float32.__module__ + "." + tf.float32.__name__ + ": <init>")
    except:
        print("tf.float32") 
    try:
        if(type(tf.argmax).__name__ == "function"): 
            function_name = tf.argmax.__name__
            print(tf.argmax.__module__ + ": " + function_name + "()")
        else: 
            print(tf.argmax.__module__ + "." + tf.argmax.__name__ + ": <init>")
    except:
        print("tf.argmax") 
    try:
        if(type(tf.nn.softmax).__name__ == "function"): 
            function_name = tf.nn.softmax.__name__
            print(tf.nn.softmax.__module__ + ": " + function_name + "()")
        else: 
            print(tf.nn.softmax.__module__ + "." + tf.nn.softmax.__name__ + ": <init>")
    except:
        print("tf.nn.softmax") 
    try:
        if(type(tf.argmax).__name__ == "function"): 
            function_name = tf.argmax.__name__
            print(tf.argmax.__module__ + ": " + function_name + "()")
        else: 
            print(tf.argmax.__module__ + "." + tf.argmax.__name__ + ": <init>")
    except:
        print("tf.argmax") 
    try:
        if(type(tf.reduce_mean).__name__ == "function"): 
            function_name = tf.reduce_mean.__name__
            print(tf.reduce_mean.__module__ + ": " + function_name + "()")
        else: 
            print(tf.reduce_mean.__module__ + "." + tf.reduce_mean.__name__ + ": <init>")
    except:
        print("tf.reduce_mean") 
    try:
        if(type(tf.nn.softmax_cross_entropy_with_logits).__name__ == "function"): 
            function_name = tf.nn.softmax_cross_entropy_with_logits.__name__
            print(tf.nn.softmax_cross_entropy_with_logits.__module__ + ": " + function_name + "()")
        else: 
            print(tf.nn.softmax_cross_entropy_with_logits.__module__ + "." + tf.nn.softmax_cross_entropy_with_logits.__name__ + ": <init>")
    except:
        print("tf.nn.softmax_cross_entropy_with_logits") 
    try:
        if(type(tf.train.GradientDescentOptimizer).__name__ == "function"): 
            function_name = tf.train.GradientDescentOptimizer.__name__
            print(tf.train.GradientDescentOptimizer.__module__ + ": " + function_name + "()")
        else: 
            print(tf.train.GradientDescentOptimizer.__module__ + "." + tf.train.GradientDescentOptimizer.__name__ + ": <init>")
    except:
        print("tf.train.GradientDescentOptimizer") 
    try:
        if(type(tf.equal).__name__ == "function"): 
            function_name = tf.equal.__name__
            print(tf.equal.__module__ + ": " + function_name + "()")
        else: 
            print(tf.equal.__module__ + "." + tf.equal.__name__ + ": <init>")
    except:
        print("tf.equal") 
    try:
        if(type(tf.reduce_mean).__name__ == "function"): 
            function_name = tf.reduce_mean.__name__
            print(tf.reduce_mean.__module__ + ": " + function_name + "()")
        else: 
            print(tf.reduce_mean.__module__ + "." + tf.reduce_mean.__name__ + ": <init>")
    except:
        print("tf.reduce_mean") 
    try:
        if(type(tf.cast).__name__ == "function"): 
            function_name = tf.cast.__name__
            print(tf.cast.__module__ + ": " + function_name + "()")
        else: 
            print(tf.cast.__module__ + "." + tf.cast.__name__ + ": <init>")
    except:
        print("tf.cast") 
    try:
        if(type(tf.float32).__name__ == "function"): 
            function_name = tf.float32.__name__
            print(tf.float32.__module__ + ": " + function_name + "()")
        else: 
            print(tf.float32.__module__ + "." + tf.float32.__name__ + ": <init>")
    except:
        print("tf.float32") 
    try:
        if(type(tf.global_variables_initializer).__name__ == "function"): 
            function_name = tf.global_variables_initializer.__name__
            print(tf.global_variables_initializer.__module__ + ": " + function_name + "()")
        else: 
            print(tf.global_variables_initializer.__module__ + "." + tf.global_variables_initializer.__name__ + ": <init>")
    except:
        print("tf.global_variables_initializer") 
    try:
        if(type(tf.train.Saver).__name__ == "function"): 
            function_name = tf.train.Saver.__name__
            print(tf.train.Saver.__module__ + ": " + function_name + "()")
        else: 
            print(tf.train.Saver.__module__ + "." + tf.train.Saver.__name__ + ": <init>")
    except:
        print("tf.train.Saver") 
    try:
        if(type(tf.Session).__name__ == "function"): 
            function_name = tf.Session.__name__
            print(tf.Session.__module__ + ": " + function_name + "()")
        else: 
            print(tf.Session.__module__ + "." + tf.Session.__name__ + ": <init>")
    except:
        print("tf.Session") 
