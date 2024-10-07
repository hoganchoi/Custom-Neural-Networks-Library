## This module stores all the loss functions for neural network models.

## Import necessary packages.
import numpy as np
import cupy as cp

##########################################################################################################################################################################
## Neural Network loss functions created using NumPy.
## Matrix operations are compiled on CPU.
##########################################################################################################################################################################

## A class that represents the Mean Absolute Error compiled on the CPU.
class MAE():
    '''
    A class that computes the Mean Absolute Error between the true value and the predicted value.

    Attributes:
        actual_value (np.array): An array containing all the true values.
        predictions (np.array): An array containing all the predictions from the model.

    Methods:
        calculateLoss (self): Calculates the error between the true value and the model's predictions.
        calculateLossGrad (self): Calculates the gradient of the error between the true value and the model's predictions.

    Usage:
        A method to calculate the loss between the model and the actual values.
    '''
    ## Initializes the MAE function on the CPU.
    def __init__(self, true_outputs, model_outputs):
        '''
        Initializes the parameters for MAE function.

        Args:
            true_outputs (np.array): An array containing all the true values.
            model_outputs (np.array): An array containing all the model's predictions for each batch.

        Returns:
            None
        '''
        self.actual_value = true_outputs
        self.predictions = model_outputs

    ## Calculates the loss between the true value and the model's predictions on CPU.
    def calculateLoss(self):
        '''
        Returns the average loss between the actual values and the model's predictions.

        Args:
            None

        Returns:
            mae (np.array): An array storing all the Mean Absolute Errors for each batch.
        '''
        ## Calculate the Mean Absolute Error between the actual value and the model's predictions.
        mae = np.mean(np.abs(self.actual_value - self.predictions), axis = 1)
        ## Return the MAE value.
        return mae
    
    ## Computes the gradient of the MAE loss function.
    def calculateLossGrad(self):
        '''
        Returns the gradient of the loss function compiled on CPU.

        Args:
            None

        Returns:
            nabla_mae (np.array): An array storing all the gradient error values for each batch.
        '''
        ## Find the number of elements in the actual value vector.
        num_elements = self.actual_value.shape[0]
        ## Calculate the gradient of the MAE function.
        nabla_mae = (-1 / num_elements) * (np.sign(self.actual_value - self.predictions))
        ## Return the gradient of MAE.
        return nabla_mae
