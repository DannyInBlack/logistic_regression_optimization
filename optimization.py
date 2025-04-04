import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import time


class LogisticRegressionBase:
    def __init__(
        self, lr=0.001, n_iters=100000, plot=True, graph_r=100, loss_limit=0.05
    ):
        self.plot = plot  # want a plot?
        self.loss_limit = loss_limit  # stops training once loss is below the limit
        self.lr = lr  # learning rate
        self.n_iters = n_iters + 1 # number of epochs
        self.ur = graph_r  # how frequently should the graph be updated
        self.weights = None  # initialize weights with nothing
        self.bias = None  # initalize biases with nothing
        self.losses = []  # initalize losses with nothing

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_predictions)
        class_preds = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_preds


class LogisticRegressionVanilla(LogisticRegressionBase):
    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape

        # initialize weights and bias with zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        if self.plot:
            plt.ion()
            _, axes = plt.subplots()
            axes.set_title("Loss During Training")
            axes.set_xlabel("Iterations")
            axes.set_ylabel("Loss")
            (line,) = axes.plot([], [], label="Loss")
            axes.legend()

        # update the after each iteration
        for i in range(self.n_iters):
            linear_predictions = np.dot(x, self.weights) + self.bias
            predictions = self.sigmoid(linear_predictions)

            dw = (1 / n_samples) * np.dot(x.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % self.ur == 0:
                # binary cross entropy
                loss = -(1 / n_samples) * np.sum(
                    y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
                )
                if self.plot: # update the graph
                    self.losses.append(loss)
                    line.set_data(
                        range(0, len(self.losses) * self.ur, self.ur), self.losses
                    )
                    axes.relim()
                    axes.autoscale_view()
                    plt.pause(0.01)

                if loss <= self.loss_limit:
                    break

        if self.plot:
            plt.ioff()


class LogisticRegressionStochastic(LogisticRegressionBase):
    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        if self.plot:
            plt.ion()
            _, axes = plt.subplots()
            axes.set_title("Loss During Training")
            axes.set_xlabel("Iterations")
            axes.set_ylabel("Loss")
            (line,) = axes.plot([], [], label="Loss")
            axes.legend()

        # update weights for each sample in every iteration
        for i in range(self.n_iters):
            for idx in range(n_samples): # do for each sample
                xi = x[idx].reshape(1, -1)
                yi = y[idx]

                linear_prediction = np.dot(xi, self.weights) + self.bias
                prediction = self.sigmoid(linear_prediction)

                dw = np.dot(xi.T, (prediction - yi))
                db = prediction - yi

                self.weights -= self.lr * dw.flatten()
                self.bias -= self.lr * db

            if i % self.ur == 0:
                linear_predictions = np.dot(x, self.weights) + self.bias
                predictions = self.sigmoid(linear_predictions)
                loss = -(1 / n_samples) * np.sum(
                    y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
                )
                if self.plot:
                    self.losses.append(loss)
                    line.set_data(
                        range(0, len(self.losses) * self.ur, self.ur), self.losses
                    )
                    axes.relim()
                    axes.autoscale_view()
                    plt.pause(0.01)

                if loss <= self.loss_limit:
                    break
        if self.plot:
            plt.ioff()


class LogisticRegressionMiniBatch(LogisticRegressionBase):
    def fit(self, x: np.ndarray, y: np.ndarray, batch_size=32):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        if self.plot:
            plt.ion()
            _, axes = plt.subplots()
            axes.set_title("Loss During Training")
            axes.set_xlabel("Iterations")
            axes.set_ylabel("Loss")
            (line,) = axes.plot([], [], label="Loss")
            axes.legend()

        # divide dataset into batches on each iteration
        for i in range(self.n_iters):
            # shuffle dataset
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]

            for start_idx in range(0, n_samples, batch_size):
                # divide into mini-batches
                end_idx = start_idx + batch_size
                xb = x[start_idx:end_idx]
                yb = y[start_idx:end_idx]

                linear_predictions = np.dot(xb, self.weights) + self.bias
                predictions = self.sigmoid(linear_predictions)

                dw = (1 / len(yb)) * np.dot(xb.T, (predictions - yb))
                db = (1 / len(yb)) * np.sum(predictions - yb)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            if i % self.ur == 0:
                linear_predictions = np.dot(x, self.weights) + self.bias
                predictions = self.sigmoid(linear_predictions)
                loss = -(1 / n_samples) * np.sum(
                    y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
                )
                if self.plot:
                    self.losses.append(loss)
                    line.set_data(
                        range(0, len(self.losses) * self.ur, self.ur), self.losses
                    )

                    axes.relim()
                    axes.autoscale_view()
                    plt.pause(0.01)

                if loss <= self.loss_limit:
                    break

        if self.plot:
            plt.ioff()


if __name__ == "__main__":
    # Load dataset and preprocess
    bc = datasets.load_breast_cancer()
    x = bc.data
    y = bc.target

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # clf = LogisticRegressionVanilla(n_iters=1000, lr=0.001)
    # clf = LogisticRegressionStochastic(n_iters=1000, lr=0.001)
    clf = LogisticRegressionMiniBatch(n_iters=1000, lr=0.001)

    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()

    plt.show()

    # evaluation
    training_time = end_time - start_time  # Calculate elapsed time
    print(f"Training Time: {training_time:.4f} seconds")

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=bc.target_names))
