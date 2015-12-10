#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    for pred, net, age in zip(predictions, net_worths, ages):
        pred, net, age = pred[0], net[0], age[0]
        error = (pred - net)**2
        tup = (age, net, error)
        cleaned_data.append(tup)

    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])
    cleaned_data = cleaned_data[:int(len(cleaned_data)*0.9)]

    return cleaned_data

