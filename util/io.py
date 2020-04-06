import csv
import numpy as np

def read_data(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) #skip header
        i = 0  
        x = []
        angle = []
        x_acc = [0]
        angle_acc = [0]
        actions = []
        states = []
        for row in reader:
            x += [(float(row[1]) - 320) / 1000] # center the 190 position at 0 and convert mm to m
            angle += [float(row[3]) * np.pi / 180] # angle in radians
            actions += [-float(row[5])]
            if angle[i] >= np.pi:
                angle[i] -= 2 * np.pi
            if len(x) > 0:
                x_acc += [x[i] - x[i-1]]
                angle_acc += [angle[i] - angle[i-1]]
                if angle_acc[i] >= np.pi:
                    angle_acc[i] -= 2 * np.pi
                if angle_acc[i] < - np.pi:
                    angle_acc[i] += 2 * np.pi
            states += [[x[i], x_acc[i], angle[i], angle_acc[i]]]
            i += 1
    return states, actions
