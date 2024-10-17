# Install required libraries
!pip install h2o tensorflow opencv-python-headless yfinance gym stable-baselines3
!apt-get install -y python3-opencv

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
import os

# Paths for chart images stored in Google Drive
base_path = '/content/drive/MyDrive/candeldata'
os.makedirs(base_path + '1min', exist_ok=True)
os.makedirs(base_path + '5min', exist_ok=True)
os.makedirs(base_path + '15min', exist_ok=True)

# Check files in Google Drive
os.listdir(base_path)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense

IMG_SIZE = (128, 128)

def create_cnn(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=output)

input_shape = (128, 128, 3)
model_1min = create_cnn(input_shape)
model_5min = create_cnn(input_shape)
model_15min = create_cnn(input_shape)

model_1min.summary()

import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define base path to your dataset
base_path = '/content/drive/MyDrive/candeldata/'

# Create dummy subfolder and move images into it
for interval in ['1min', '5min', '15min']:
    class_a_dir = os.path.join(base_path, interval, 'classA')  # Create a 'classA' folder
    os.makedirs(class_a_dir, exist_ok=True)

    # Move all images from the main folder into 'classA'
    for filename in os.listdir(os.path.join(base_path, interval)):
        if filename.endswith('.png'):  # Ensure we're moving only PNG images
            file_path = os.path.join(base_path, interval, filename)
            shutil.move(file_path, class_a_dir)

# --- Step 4: Image Preprocessing and Data Loading ---

# Define image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Create an instance of ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load 1min images
train_gen_1min = datagen.flow_from_directory(
    base_path + '1min',  # Path to the '1min' folder
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Since we're treating all images as belonging to one class
    subset='training'
)
val_gen_1min = datagen.flow_from_directory(
    base_path + '1min',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load 5min images
train_gen_5min = datagen.flow_from_directory(
    base_path + '5min',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_gen_5min = datagen.flow_from_directory(
    base_path + '5min',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load 15min images
train_gen_15min = datagen.flow_from_directory(
    base_path + '15min',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_gen_15min = datagen.flow_from_directory(
    base_path + '15min',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- Now you can use train_gen_1min, val_gen_1min, train_gen_5min, val_gen_5min, train_gen_15min, and val_gen_15min for model training ---

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# --- Step 1: Data Loading from Google Drive Folders ---

base_path = '/content/drive/MyDrive/candeldata/'  # Path to your dataset

# Ensure the correct folder structure
os.makedirs(base_path + '1min', exist_ok=True)
os.makedirs(base_path + '5min', exist_ok=True)
os.makedirs(base_path + '15min', exist_ok=True)

# Image size
IMG_SIZE = (128, 128)

# Create data generator for training and validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load images from 1min, 5min, and 15min directories
train_gen_1min = datagen.flow_from_directory(base_path + '1min', target_size=IMG_SIZE, subset='training', batch_size=32, class_mode='binary')
val_gen_1min = datagen.flow_from_directory(base_path + '1min', target_size=IMG_SIZE, subset='validation', batch_size=32, class_mode='binary')

train_gen_5min = datagen.flow_from_directory(base_path + '5min', target_size=IMG_SIZE, subset='training', batch_size=32, class_mode='binary')
val_gen_5min = datagen.flow_from_directory(base_path + '5min', target_size=IMG_SIZE, subset='validation', batch_size=32, class_mode='binary')

train_gen_15min = datagen.flow_from_directory(base_path + '15min', target_size=IMG_SIZE, subset='training', batch_size=32, class_mode='binary')
val_gen_15min = datagen.flow_from_directory(base_path + '15min', target_size=IMG_SIZE, subset='validation', batch_size=32, class_mode='binary')

# --- Step 2: Define CNN Model ---
def create_cnn_model(input_shape):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))

    # Dropout layer
    model.add(Dropout(0.5))

    # Output layer (since it's binary classification)
    model.add(Dense(1, activation='sigmoid'))

    return model

# Input shape for the images (128x128 with 3 color channels)
input_shape = (128, 128, 3)

# Create the models for 1min, 5min, and 15min data
model_1min = create_cnn_model(input_shape)
model_5min = create_cnn_model(input_shape)
model_15min = create_cnn_model(input_shape)

# --- Step 3: Compile the Models ---
# Since it's binary classification, use 'binary_crossentropy'
model_1min.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_5min.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_15min.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Step 4: Train the Models ---

# Train the model for 1min data
history_1min = model_1min.fit(train_gen_1min, validation_data=val_gen_1min, epochs=5)

# Train the model for 5min data
history_5min = model_5min.fit(train_gen_5min, validation_data=val_gen_5min, epochs=5)

# Train the model for 15min data
history_15min = model_15min.fit(train_gen_15min, validation_data=val_gen_15min, epochs=5)

# --- Step 5: Evaluate the Models ---

# Evaluate 1min model on validation data
val_loss_1min, val_acc_1min = model_1min.evaluate(val_gen_1min)
print(f"1min Model - Validation Loss: {val_loss_1min}, Validation Accuracy: {val_acc_1min}")

# Evaluate 5min model on validation data
val_loss_5min, val_acc_5min = model_5min.evaluate(val_gen_5min)
print(f"5min Model - Validation Loss: {val_loss_5min}, Validation Accuracy: {val_acc_5min}")

# Evaluate 15min model on validation data
val_loss_15min, val_acc_15min = model_15min.evaluate(val_gen_15min)
print(f"15min Model - Validation Loss: {val_loss_15min}, Validation Accuracy: {val_acc_15min}")

# --- Step 6: Visualize Training History ---

# Function to plot training and validation accuracy/loss
def plot_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title(f'{title} Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'{title} Loss')
    plt.legend()

    plt.show()

# Plot for each model
plot_history(history_1min, '1min Model')
plot_history(history_5min, '5min Model')
plot_history(history_15min, '15min Model')
# --- Step 5: Evaluate the Models ---

# Evaluate 1min model on validation data
val_loss_1min, val_acc_1min = model_1min.evaluate(val_gen_1min)
print(f"1min Model - Validation Loss: {val_loss_1min:.3f}, Validation Accuracy: {val_acc_1min * 100:.2f}%")

# Evaluate 5min model on validation data
val_loss_5min, val_acc_5min = model_5min.evaluate(val_gen_5min)
print(f"5min Model - Validation Loss: {val_loss_5min:.3f}, Validation Accuracy: {val_acc_5min * 100:.2f}%")

# Evaluate 15min model on validation data
val_loss_15min, val_acc_15min = model_15min.evaluate(val_gen_15min)
print(f"15min Model - Validation Loss: {val_loss_15min:.3f}, Validation Accuracy: {val_acc_15min * 100:.2f}%")

import os
from skimage import io, color, feature
import numpy as np

# Define the base directory where your image folders are located
base_dir = '/content/drive/MyDrive/candeldata'

# List the subfolders (e.g., '15min', '1min', '5min') containing images
folders = ['15min', '1min', '5min']

# Initialize a list to store the features for each image
image_features = []

# Iterate through each folder
for folder in folders:
    folder_path = os.path.join(base_dir, folder)

    # List all PNG files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Process each image in the folder
    for image_file in image_files:
        # Build the full image path
        image_path = os.path.join(folder_path, image_file)

        # Load the image
        img = io.imread(image_path)

        # Convert to grayscale
        gray_img = color.rgb2gray(img)

        # Extract edge features using Canny edge detection
        edges = feature.canny(gray_img)

        # Flatten the edge features to create a feature vector
        feature_vector = edges.flatten()

        # Append the feature vector along with the folder and image name (for identification)
        image_features.append({
            'folder': folder,
            'image_file': image_file,
            'features': feature_vector
        })

# At this point, image_features contains the extracted features for all images in all folders
# You can now use these features for training a machine learning model

# Example: Convert the list of features to a NumPy array for easier manipulation
features_array = np.array([item['features'] for item in image_features])

# Now you can use `features_array` as the input data for your machine learning model

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the neural network model (Q-network)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer to store experiences
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

# DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.update_target_steps = 1000
        self.steps = 0

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Random action
        else:
            state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()  # Choose action with highest Q-value

    def train(self):
        """Train the Q-network with experiences from the replay buffer."""
        if self.replay_buffer.size() < self.batch_size:
            return  # Not enough experience to train

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q-values of the current states
        q_values = self.q_network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q-values of the next states (using target network)
        next_q_values = self.target_network(next_states)
        max_next_q = next_q_values.max(1)[0]
        expected_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute loss and update the Q-network
        loss = self.loss_fn(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network every few steps
        if self.steps % self.update_target_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def update_epsilon(self):
        """Decrease epsilon for exploration-exploitation tradeoff."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Main loop to run the DQN algorithm
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    episodes = 500
    max_steps_per_episode = 500

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.add_experience(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        # Decrease epsilon for exploration-exploitation tradeoff
        agent.update_epsilon()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    env.close()
import numpy as np

# Function to detect upward or downward trends
def detect_trend(data):
    """
    Detect upward or downward trends based on price changes.
    Returns 'up' for upward trends, 'down' for downward trends.
    """
    return np.where(np.diff(data) > 0, 'up', 'down')

# Function to detect a simple moving average crossover (bullish/bearish)
def moving_average_crossover(data, short_window=3, long_window=5):
    """
    Detects moving average crossovers (bullish/bearish).
    - Bullish crossover: short-term moving average crosses above the long-term.
    - Bearish crossover: short-term moving average crosses below the long-term.
    """
    short_ma = np.convolve(data, np.ones(short_window)/short_window, mode='valid')
    long_ma = np.convolve(data, np.ones(long_window)/long_window, mode='valid')

    # Compare short MA and long MA to detect crossovers
    signals = np.where(short_ma[-len(long_ma):] > long_ma, 'bullish', 'bearish')
    return signals

# Function to detect a bullish or bearish engulfing pattern (simplified version)
def detect_engulfing_pattern(data):
    """
    Detects bullish or bearish engulfing candlestick patterns.
    - Bullish engulfing: Downtrend followed by a larger up-candle.
    - Bearish engulfing: Uptrend followed by a larger down-candle.
    """
    pattern = []
    for i in range(1, len(data) - 1):
        if data[i-1] > data[i] and data[i+1] > data[i]:
            pattern.append('bullish_engulfing')
        elif data[i-1] < data[i] and data[i+1] < data[i]:
            pattern.append('bearish_engulfing')
        else:
            pattern.append('no_pattern')
    return pattern

# Example price data (representing closing prices of a stock)
prices = [100, 102, 101, 105, 108, 107, 109, 111, 110, 112]

# Detect trends (upward or downward)
trend = detect_trend(prices)
print("Trend Detection:")
print(trend)

# Detect moving average crossover signals
ma_signals = moving_average_crossover(prices, short_window=3, long_window=5)
print("\nMoving Average Crossover Detection:")
print(ma_signals)

# Detect engulfing patterns
engulfing_patterns = detect_engulfing_pattern(prices)
print("\nBullish/Bearish Engulfing Pattern Detection:")
print(engulfing_patterns)
!pip install deap
# Step 0: Install DEAP if not already installed
!pip install deap

# Import necessary modules from DEAP and Python's standard library
from deap import base, creator, tools, algorithms
import random

# Step 1: Define the Individual and Fitness function
# The FitnessMax maximizes the objective (weights=(1.0,)).
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# The Individual is a list of attributes with a corresponding fitness score
creator.create("Individual", list, fitness=creator.FitnessMax)

# Step 2: Define the population, crossover, mutation, and selection
toolbox = base.Toolbox()
# Attribute generator: Generate a random float in the range [0.0, 1.0)
toolbox.register("attr_float", random.random)
# Structure initializers: Define an individual (a list of 5 random floats)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
# Define the population as a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Step 3: Define the fitness function (the goal is to maximize the sum of the individual's values)
def evaluate(individual):
    return sum(individual),  # Return a tuple (sum of individual values)

# Step 4: Register the GA operations
toolbox.register("evaluate", evaluate)  # Fitness function
toolbox.register("mate", tools.cxTwoPoint)  # Crossover method: two-point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Mutation method: flip a bit with probability 0.05
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection method: tournament selection

# Step 5: Set up the Genetic Algorithm process
# Generate the initial population of 100 individuals
population = toolbox.population(n=100)

# Step 6: Evolve the population using the genetic algorithm
# Run the genetic algorithm for 40 generations
# cxpb = crossover probability, mutpb = mutation probability
result_population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=True)

# Step 7: Extract and display the best individual from the final population
best_individual = tools.selBest(result_population, 1)[0]
print(f"\nBest individual is: {best_individual}")
print(f"Fitness: {evaluate(best_individual)[0]}")
import matplotlib.pyplot as plt

# Example data for predictions
actual = [10, 12, 15, 18, 16, 14]       # Actual values
predicted = [11, 13, 14, 17, 15, 13]    # Predicted values

# Create a new figure for plotting
plt.figure(figsize=(10, 5))  # Set the figure size

# Plot actual vs predicted values
plt.plot(actual, label="Actual", marker='o')        # Actual values with markers
plt.plot(predicted, label="Predicted", marker='x')  # Predicted values with different markers

# Add titles and labels
plt.title("Actual vs Predicted Values")
plt.xlabel("Time")
plt.ylabel("Values")

# Add a legend to differentiate actual and predicted values
plt.legend()

# Add a grid for better readability
plt.grid()

# Show the plot
plt.show()
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load a sample dataset (Iris dataset in this example)
data = load_iris()
X = data.data  # Features
y = data.target  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the individual models
clf1 = RandomForestClassifier(random_state=42)  # Random Forest model
clf2 = LogisticRegression(max_iter=200)          # Logistic Regression model
clf3 = SVC(probability=True, random_state=42)    # Support Vector Classifier model

# Create the ensemble model using Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('rf', clf1),
    ('lr', clf2),
    ('svc', clf3)
], voting='hard')

# Fit the ensemble model on the training data
ensemble_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ensemble_model.predict(X_test)

# Calculate and print the accuracy of the ensemble model
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble model accuracy: {accuracy:.2f}")
# Import necessary libraries
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Load and preprocess the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Step 2: Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,               # Randomly rotate images in the range (degrees, 0 to 20)
    width_shift_range=0.2,           # Randomly translate images horizontally (20% of total width)
    height_shift_range=0.2,          # Randomly translate images vertically (20% of total height)
    shear_range=0.2,                 # Shear intensity (shear angle in counter-clockwise direction in degrees)
    zoom_range=0.2,                  # Randomly zoom into images (20% of the image)
    horizontal_flip=True,            # Randomly flip images horizontally
    fill_mode='nearest'              # Fill in newly created pixels after a transformation
)

# Step 3: Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Fit the model using the data generator
# Use datagen.flow to augment images during training
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))
# Import necessary libraries
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Step 1: Load your dataset
# For demonstration, let's create a synthetic dataset (replace this with your dataset)
# Assuming you have images of shape (128, 128, 3) and binary labels
num_samples = 1000
X = np.random.rand(num_samples, 128, 128, 3)  # Random images
y = np.random.randint(0, 2, num_samples)      # Random binary labels

# Step 2: Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Step 4: Load the VGG16 pre-trained model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Step 5: Add custom layers on top of the base model
x = Flatten()(base_model.output)  # Flatten the output of the base model
x = Dense(128, activation='relu')(x)  # Add a dense layer with 128 units
predictions = Dense(1, activation='sigmoid')(x)  # Add an output layer for binary classification

# Step 6: Create the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Step 7: Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])  # Use learning_rate instead of lr

# Step 8: Fit the model using the data generator
# Use datagen.flow for data augmentation during training
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_val, y_val),
          epochs=10)
# Import necessary libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification

# Step 1: Load or create your dataset
# For demonstration, let's create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Step 2: Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the model
model = RandomForestClassifier()

# Step 4: Set up the hyperparameter grid
params = {
    'n_estimators': [50, 100],     # Number of trees in the forest
    'max_depth': [10, 20, None]    # Maximum depth of the tree
}

# Step 5: Set up the GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)  # 5-fold cross-validation

# Step 6: Fit the model using grid search
grid_search.fit(X_train, y_train)

# Step 7: Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Hyperparameters:", best_params)
print("Best Cross-validation Score:", best_score)
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification

# Step 1: Load or create your dataset
# For demonstration, let's create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Step 2: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Step 6: Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:\n", cm)
