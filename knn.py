import math

import numpy as np


def knn(X_train, X_test, Y_train, Y_test, k=5):

    predictions = np.zeros(shape=Y_test.shape)

    for i in range(len(X_test)):

        row = X_test[i, :]

        closest = []
        for j in range(len(X_train)):

            point = X_train[j, :]
            dist = np.linalg.norm(row - point)
            if len(closest) < k:
                closest.append( (Y_train[j], dist) )
            else:
                furthest_curr = closest[-1][1] # See if we can replace this
                if dist < furthest_curr:
                    closest.pop(-1)
                    closest.append( (Y_train[j], dist) )

        # Now that we have closest populated correctly, we can classify the row

        to_choose = {}
        for label, _ in closest:
            if label in to_choose:
                to_choose[label] = to_choose[label] + 1
            else:
                to_choose[label] = 1

        predicted_label = max(to_choose, key=to_choose.get)
        predictions[i] = predicted_label

    diff = predictions - Y_test
    total_correct = np.sum(diff == 0)
    # print(str(k),"&", round( (total_correct / len(Y_test)), 3), "\\\\") # Print the losses
    return round( (total_correct / len(Y_test)), 6) # Return the testing loss










