import csv
import numpy as np

def read_data(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) #skip header
        x = 0
        last_x = 0
        last_angle = 0
        angle = 0
        x_dot = 0
        angle_dot = 0
        actions = []
        states = []
        for row in reader:
            x = (float(row[1]) - 320.) / 1000. # center the 190 position at 0 and convert mm to m
            angle = float(row[3]) * np.pi / 180. # angle in radians
            actions += [-float(row[5])]
            if angle >= np.pi:
                angle -= 2 * np.pi
            x_dot = (x - last_x) / 0.02
            angle_dot = (angle - last_angle)
            if angle_dot >= np.pi:
                angle_dot -= 2 * np.pi
            if angle_dot < - np.pi:
                angle_dot += 2 * np.pi
            angle_dot /= 0.02
            states += [[x, x_dot, angle, angle_dot]]
            last_x = x
            last_angle = angle
    return states, actions
