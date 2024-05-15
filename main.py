import pandas as pd
import numpy as np
from knn import knn
import torch
from matplotlib.lines import Line2D
from promise.dataloader import DataLoader
from models import (Linear, test, train, correct_predict_num, Nonlinear_Net)
from torch import nn, optim

from torch.utils.data import DataLoader, Dataset
import matplotlib.patches as mpatches

from utils import PremDataset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# Preprocesses the columns of the dataframe. Returns X, Y where X is 42 columns and Y is labels (0, 1, 2)
# do_pca = True returns X with only 3 columns
# show_plot = True shows the PCA plot
def preprocessing(do_pca=False, show_plot=False):
    file_path = "dataset.csv"
    df = pd.read_csv(file_path)

    df = df[df["Position"] != "Goalkeeper"]

    df.loc[df['Goals per match'] == df['Appearances'], "Goals per match"] = 0

    delete_columns = ["Name", "Jersey Number", "Club", "Nationality",
                      "Age", "Appearances", "Wins", "Losses", "Saves", "Penalties saved", "Punches",
                      "High Claims", "Catches", "Sweeper clearances", "Throw outs", "Goal Kicks"]

    df = df.drop(columns=delete_columns)

    df = df.fillna(0)

    df['Shooting accuracy %'] = df['Shooting accuracy %'].replace(0, '0%')
    df['Tackle success %'] = df['Tackle success %'].replace(0, '0%')
    df['Cross accuracy %'] = df['Cross accuracy %'].replace(0, '0%')

    df['Shooting accuracy %'] = df['Shooting accuracy %'].str.replace('%', '').astype(int)
    df['Tackle success %'] = df['Tackle success %'].str.replace('%', '').astype(int)
    df['Cross accuracy %'] = df['Cross accuracy %'].str.replace('%', '').astype(int)

    df = df.fillna(0)

    position_mapping = {'Defender': 0, 'Midfielder': 1, 'Forward': 2}
    df['Position'] = df['Position'].replace(position_mapping)

    Y = df['Position'].values
    X = df.drop(columns=['Position']).values

    Y = np.array(Y)
    X = np.array(X)

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Normalize features

    ### PCA ###
    pca = PCA(n_components=3).fit(X)
    X_reduced = pca.transform(X)
    ### PCA ###

    ### PLOTTING ###
    ax = plt.figure().add_subplot(projection='3d')

    colors = []
    for i in Y:
        if i == 0:
            colors.append("blue")
        elif i == 1:
            colors.append("green")
        else:
            colors.append("red")

    x = X_reduced[:, 0]
    y = X_reduced[:, 1]
    z = X_reduced[:, 2]
    ax.scatter(x, y, z, alpha=0.9, color=colors)
    plt.title('3D PCA of Dataset')
    handles = [mpatches.Patch(color="blue", label="Defender"),
               mpatches.Patch(color="green", label="Midfielder"),
               mpatches.Patch(color="red", label="Offense")]
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.legend(handles=handles, bbox_to_anchor=(1, 1), loc="upper right")

    if show_plot:
        plt.show()  # Shows plot of reduced dimension data. Can comment out.

    ### PLOTTING ###

    ###### Apply PCA to return value #######
    if do_pca:
        X = X_reduced
    ###### Apply PCA to return value #######

    return X, Y


# Converts processed data into dataloaders to be passed into test and train functions for neural networks.
def convert_to_dataloader(X_train, Y_train, X_test, Y_test, batch_size=8):
    # Build dataset
    dataset_train = PremDataset(X_train, Y_train)
    dataset_test = PremDataset(X_test, Y_test)

    # Build dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
    return dataloader_train, dataloader_test


# Creates either a Nonlinear or Linear model, where hidden_size is optional
def create_model(model_type, input_size, hidden_size=50):
    # batch_size = 8  # batch size (optimal so far 8)
    global model

    if model_type == "Nonlinear":
        model = Nonlinear_Net(input_size=input_size, hidden_size=hidden_size, num_classes=3)
    elif model_type == "Linear":
        model = Linear(input_features=input_size)
    else:
        print("model_type must be Nonlinear or Linear")
    return model


# Runs a model given appropriate dataloaders, epochs, and learning rate
def run_model(dataloader_train, dataloader_test, model, num_epoch=5, learning_rate=0.01):
    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses, accuracies = train(
        model,
        dataloader_train,
        loss_func,
        optimizer,
        num_epoch,
        correct_num_func=correct_predict_num,
    )

    loss_test, accuracy_test = test(
        model, dataloader_test, loss_func, correct_num_func=correct_predict_num
    )

    ### PRINT STATS IF DESIRED ###
    # print("Average Training Loss and Accuracy:", np.mean(losses), np.mean(accuracies))
    # print("Final Training Accuracy:", accuracies[-1])
    # print("Test Accuracy:", accuracy_test)
    ### PRINT STATS IF DESIRED ###

    return accuracy_test


# To run a custom example after the model has been trained already
# example row = [3, 2, 2] after using PCA to reduce to 3 Dimensions
def predict_custom_example(row, model):
    row = torch.tensor(row, dtype=torch.float32)

    row = row.unsqueeze(0)  # We must add a batch dimension (we want [1,3] not just [3], for example with reduced data)

    # Put the model in evaluation mode
    model.eval()

    # Forward pass to get the prediction
    with torch.no_grad():
        output = model(row)

    # Convert the output to probabilities using softmax
    probabilities = torch.softmax(output, dim=1)
    prob_vals = probabilities.detach().numpy()[0]
    print("\n" + str(prob_vals[0]) + "% Defender", str(prob_vals[1]) + " % Midfielder",
          str(prob_vals[2]) + " % Offense")

    # Get the predicted class (the one with the highest probability)
    predicted_class = torch.argmax(probabilities, dim=1)

    print("Predicted Class:", predicted_class.item())


# Generates index groups based on amount of groups needed and dataset (random permutations).
def split_indices(dataset, k):
    num_data = dataset.shape[0]
    fold_size = int(num_data / k)
    indices = np.random.permutation(num_data)
    indices_split = np.split(indices[: fold_size * k], k)
    return indices_split


# Runs K-Fold validation for Neural Network Models. Takes in a param_list for hyperparameters.
def runKFold(X, Y, input_size, k=4, param_list=None):
    if param_list:
        model_type = param_list[0]
        num_epochs = param_list[1]
        learning_rate = param_list[2]
        batch_size = param_list[3]
        hidden_size = param_list[4]

    groups = split_indices(X, k)
    cross_val_loss = 0
    for i in range(k):
        test_indices = groups[i]
        X_test = X[test_indices]
        Y_test = Y[test_indices]

        train_indices = list(range(len(X)))
        for i in test_indices:
            train_indices.remove(i)

        X_train = X[train_indices]
        Y_train = Y[train_indices]

        #### RUN THE MODEL AND UPDATE cross_val_loss ####

        # Build dataset

        create_model(model_type=model_type, input_size=input_size, hidden_size=hidden_size)
        dataset_train = PremDataset(X_train, Y_train)
        dataset_test = PremDataset(X_test, Y_test)

        # Build dataloader
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

        test_accuracy = run_model(dataloader_train, dataloader_test, model, learning_rate=learning_rate,
                                  num_epoch=num_epochs)
        cross_val_loss += 1 - test_accuracy
        #### RUN THE MODEL AND UPDATE cross_val_loss ####

    cross_val_loss = cross_val_loss / k  # average it out
    cross_val_loss = round(cross_val_loss, 3)
    print(param_list[0], "&", param_list[1], "&", param_list[2], "&", param_list[3], "&", param_list[4], "&",
          cross_val_loss, "\\\\")
    return cross_val_loss


# Runs KFold validation for KNN model
def KNN_KFold(X, Y, k_splits=4, k_neighbors=1):
    groups = split_indices(X, k_splits)
    cross_val_loss = 0
    for i in range(k_splits):
        test_indices = groups[i]
        X_test = X[test_indices]
        Y_test = Y[test_indices]

        train_indices = list(range(len(X)))
        for i in test_indices:
            train_indices.remove(i)

        X_train = X[train_indices]
        Y_train = Y[train_indices]

        test_accuracy = knn(X_train, X_test, Y_train, Y_test, k=k_neighbors)
        cross_val_loss += 1 - test_accuracy
    cross_val_loss = cross_val_loss / k_splits
    cross_val_loss = round(cross_val_loss, 3)
    print(k_neighbors, "&", cross_val_loss, "\\\\")
    return cross_val_loss


if __name__ == '__main__':

    np.random.seed(16)  # Same random seed for consistency. Also applies to random_state of 0.

    X, Y = preprocessing(do_pca=False,show_plot=False)  # X and Y are the same data as the dataloaders but without the PremDataset/Dataloader Wrappers


    ###### Generate parameter combinations (each is a list) ######
    model_type_list = ["Nonlinear", "Linear"]
    epoch_list = [15, 75]
    learning_rate_list = [0.005, 0.01, 0.1]
    hidden_size_list = [5, 25, 75]
    batch_size_list = [2, 8]
    param_lists = []
    for m in model_type_list:
        for e in epoch_list:
            for l in learning_rate_list:
                for h in hidden_size_list:

                    for b in batch_size_list:
                        param_lists.append([m, e, l, b, h])
    ###### Generate parameter combinations (each is a list) ######

    X_else, X_test, Y_else, Y_test = train_test_split(X, Y, test_size=0.2,
                                                      random_state=0)  # Split into two sets, A and B (A is testing, B contains training and validation data that will be indexed in different combinations)

    ### RUN VALIDATION SPLITS ###
    # for o in param_lists:
    #     u = runKFold(X, Y, 42 , param_list=o)
    ### RUN VALIDATION SPLITS ###

    #### TEST MOST OPTIMAL NEURAL NET MODEL ####

    # most_optimal_model = create_model("Nonlinear", input_size=42, hidden_size=75)
    # dataloader_train, dataloader_test = convert_to_dataloader(X_else, Y_else, X_test, Y_test, batch_size=2)
    # test_acc = run_model(dataloader_train, dataloader_test, most_optimal_model, num_epoch=75, learning_rate=0.005)
    # print(test_acc)

    #### TEST MOST OPTIMAL NEURAL NET MODEL ####

    ### RUN CROSS VALIDATION FOR KNN ####
    # for k in [1,2,3,5,10,25,50,100]:
    #     KNN_KFold(X_else, Y_else, k_neighbors=k)
    # best_knn_acc = knn(X_else, X_test, Y_else, Y_test, k=1)
    # print(best_knn_acc)
    ### RUN CROSS VALIDATION FOR KNN ####

    ### an example for predict_custom_example in original space ###
    example_X = [150, 0.000e+00, 0.000e+00, 4.000e+00, 3.000e+00, 0.000e+00, 0.000e+00,
                 0.000e+00, 0.000e+00, 0.000e+00, 3.000e+00, 0.000e+00, 5.300e+01, 1.660e+02,
                 2.140e+02, 7.800e+01, 1.000e+00, 3.200e+01, 2.080e+02, 3.040e+02, 1.430e+02,
                 3.000e+00, 7.320e+02, 6.110e+02, 7.090e+02, 1.960e+02, 1.610e+02, 2.150e+02,
                 1.000e+00, 1.000e+00, 1.800e+01, 7.125e+03, 4.453e+01, 2.800e+01, 3.890e+02,
                 1.600e+01, 3.100e+01, 1.440e+02, 2.300e+01, 0.000e+00, 1.250e+02, 8.000e+00]
    ### an example for predict_custom_example in original space ###
