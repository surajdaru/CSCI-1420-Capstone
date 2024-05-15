There are a few different components that can be run in this project. Each component is commented 
under "if __name__ == '__main__':" in main.py. I simply used python3 main.py inside the directory to run the program.

1. If you would like to run the cross validation splits, simply comment out lines 291-294.
2. For Running the most optimal Neural Network, or really any custom Neural Network Model, uncomment 298-301 and adjust
3. Uncomment 306-309 for KNN Cross-Validation 
4. In the preprocessing on line 269, you can choose whether or not to use PCA to reduce the dimension of the data to
    3, or keep it at 42. Also, you can choose whether or not to have the plot show. Keep in mind that by changing the 
    dimension here, you will have to adjust the "input_size" parameter when calling create_model and runKFold below.