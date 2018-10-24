1) To execute our file, type the following into the terminal:

### python3 neuralnet.py ###

This will execute the code that coincides with parts (c)-(f) of the assignment. To execute part (b) code, you will need to go into the file and uncomment the corresponding section. 

For each part (c)-(f), the file will print out the training and testing accuracies for epochs 0, 5, 10, 15, 20, 25, and 30 (30 because each model is trained based on the optimal number of epochs). Any graphs that plot the training vs. testing accuracies will be stored in the "/images" directory.

*** IMPORTANT NOTE *** 
Running this code will take awhile to execute, considering that training the network for part (c) requires an initial run of 50 epochs * 10 trials over our large data set.

2) If you wish to import the file as a module, then you need to uncomment the "main" section, and the following will be imported: 

  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'
  
  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config) 
