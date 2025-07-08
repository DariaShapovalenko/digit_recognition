import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from neural_net import NeuralNetwork, normalize

def load_csv(filename):
    data, labels = [], []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0])
            features = [float(x) for x in row[1:]]
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

def one_hot_encode(labels, num_classes=10):
    return np.array([[1 if i == label else 0 for i in range(num_classes)] for label in labels])

def split_data(data, labels, test_ratio=0.2):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    split = int(len(combined) * (1 - test_ratio))
    train = combined[:split]
    test = combined[split:]
    return (np.array([x[0] for x in train]), np.array([x[1] for x in train])), (np.array([x[0] for x in test]), np.array([x[1] for x in test]))

if __name__ == "__main__":
    data, labels = load_csv("data.csv")
    data = normalize(data)
    
    labels_one_hot = one_hot_encode(labels)
    
    (train_data, train_labels), (test_data, test_labels) = split_data(data, labels_one_hot)
    
    input_size = train_data.shape[1]
    
    net = NeuralNetwork(
        layer_sizes=[input_size, 128, 64, 10],
        activation='relu',
        learning_rate=0.01
    )
    
    net.train(train_data, train_labels, epochs=20, batch_size=32)
    
    accuracy = net.evaluate(test_data, test_labels) * 100
    print(f"\nАкуратність: {accuracy:.2f}%")
    
    net.save("digit_model.json")
    
    print("\n передбачення:")
    for i in range(5):
        x = test_data[i]
        true_label = np.argmax(test_labels[i])
        predicted_probs = net.predict(x)
        predicted_label = np.argmax(predicted_probs)
        print(f"приклад {i+1}: реал = {true_label}, очікуване = {predicted_label}, " 
              f"впевненість = {predicted_probs[predicted_label][0]:.4f}")
