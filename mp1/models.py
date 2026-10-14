import numpy as np
import logging
from utils import compute_accuracy


class NearestNeighbor(object):
    def __init__(self, data, labels, k):
        """
        Args:
            data: n_train x d matrix with a d-dimensional feature for each of
            the n_train points
            labels: n_train vector with the label for each of the n points
            k: number of nearest neighbors to use for prediction
        """
        self.k = k
        self.data = data
        self.labels = labels

    def train(self):
        """
        Trains the model and stores in class variables whatever is necessary to
        make predictions later.
        """
        # BEGIN YOUR CODE
        # 100 labels, each with 784 dimensions (directions)
        # self.Xtrain = self.data
        # self.labels = self.labels

        # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n_test x d matrix with a d-dimensional feature for each of the
            n_test points
        Returns:
            y: n_test vector with the predicted label for each of the n_test points
        """
        # BEGIN YOUR CODE
        # print(x)
        # .\.venv\Scripts\Activate.ps1
        num_test = x.shape[0]  # number of rows/number of items to predict
        # print("Amount of test data to predict: ", num_test)
        yprediction = np.zeros(num_test, dtype=self.labels.dtype)
        # print(yprediction)
        for i in range(num_test):
            # print(f"\nProcessing Test Point {i}: {x[i, :][0:10]}...")
            # print(x[i, :].shape)

            distances = np.sqrt(np.sum(np.square(self.data - x[i, :]), axis=1))
            # print(distances.shape)
            # print(distances)

            min_k_distances_indexes = np.argpartition(distances, self.k)[: self.k]
            # print(min_k_distances_indexes)
            # print(self.labels)
            votes = []
            for j in min_k_distances_indexes:
                # print(j)
                # print(self.labels[j])
                votes.append(self.labels[j])
            # print("Votes:", votes)
            mode = votes[0]
            max_count = 0

            for vote in votes:
                count = 0

                for v in votes:
                    if v == vote:
                        count += 1

                if count > max_count:
                    max_count = count
                    mode = vote
            # print("Mode:", mode)
            # TODO failing, cuz there is a tie in mode?
            yprediction[i] = mode
            # print(yprediction.shape)
            # print(yprediction)
        # Visualize the 10 nearest neighbors
        # for 5 random samples from the validation dataset in
        # top_imgs = self.get_nearest_neighbors(x, self.k)
        return yprediction

        # END YOUR CODE

    def get_nearest_neighbors(self, x, k):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
            k: number of nearest neighbors to return
        Returns:
            top_imgs: n x k x d vector containing the nearest neighbors in the
            training data, top_imgs must be sorted by the distance to the
            corresponding point in x.
        """
        # BEGIN YOUR CODE
        print("nearest neighbors")
        print(x.shape)
        num_sample = 5
        # pick 5 random x out of 10000, get k nearest nighbors, then visualize
        indices = np.random.choice(x.shape[0], size=num_sample, replace=False)
        print(indices)

        x = x[indices]
        print(x)
        print(x.shape)

        top_imgs = np.zeros((num_sample, k, x.shape[1]), dtype=self.data.dtype)

        for i in range(num_sample):
            distances = np.sqrt(np.sum(np.square(self.data - x[i, :]), axis=1))

            sorted_indices = np.argsort(distances)
            # [start, end , step]
            nearest_k_indices = sorted_indices[:k]

            top_imgs[i] = self.data[nearest_k_indices]
        print("top images")
        print(top_imgs.shape)
        # knn_x represents the actual 28x28 pixel data of the K-nearest neighbors found for each of your query images
        from utils import visualize_knn

        visualize_knn(x, top_imgs, file_name=None)

        return top_imgs
        # END YOUR CODE


class LogisticRegressionClassifier(object):
    def __init__(self, data, labels, epochs=10, lr=1e-3, reg_wt=3e-5, writer=None):
        self.data = data
        self.labels = labels
        self.epochs = epochs
        self.lr = lr
        self.reg_wt = reg_wt
        self.rng = np.random.RandomState(1234)
        std = 1.0 / np.sqrt(data.shape[1])
        self.w = self.rng.uniform(-std, std, size=(self.data.shape[1], 10))
        self.writer = writer

    # def softmax(self, x, w_c):
    #     scores = w_c.T @ x
    #     # trick to avoid overflow
    #     scores -= np.max(scores)
    #     exps = np.exp(scores)
    #     # Return p_c(x) for all c
    #     return exps / np.sum(exps)
    def softmax(self, x, W):
        ten_p = x @ W

        # ten_p  (N, 10) axis=1 so columns, moving across horizontally.
        ten_p -= np.max(ten_p, axis=1, keepdims=True)  # scores -= np.max(scores)
        exp_ten_p = np.exp(ten_p)  # shape: (N, 10)

        # sum accross each row
        return exp_ten_p / np.sum(exp_ten_p, axis=1, keepdims=True)

    # def L_d(self, x, y, W):
    #     N = x.shape[0]
    #     loss_sum = 0.0

    #     for i in range(N):
    #         ten_p = self.softmax(x[i], W)
    #         # Index ten_p with the scalar label for the i-th point: y[i]
    #         loss_sum += np.log(ten_p[y[i]])

    #     L_d = -1.0 * loss_sum / N

    #     return L_d

    def L_d(self, x, y, W):
        N = x.shape[0]
        ten_p = self.softmax(x, W)

        # loss_sum += np.log(ten_p[y[i]])
        # np.arange(N) shape: (N,1), y.squeeze() shape: (N,1)
        # grabs ten_p[0, y[0]], ten_p[1, y[1]], ...
        # that way only when yi and xi have the same i
        loss_sum = np.sum(np.log(ten_p[np.arange(N), y.squeeze()]))

        # L_d = -1.0 * loss_sum / N
        L_d_val = -1.0 * loss_sum / N

        return L_d_val

    def L_r(self, w_c):
        squared_norm = np.sum(w_c * w_c)
        return (self.reg_wt / 2) * squared_norm

    # def gradient_L_d_w_j(self, x, y, W, j):
    #     # fix w and j

    #     N = x.shape[0]
    #     # print(x.shape)
    #     grad_sum = np.zeros_like(x[0])

    #     for i in range(N):
    #         ten_p = self.softmax(x[i], W)

    #         kronecker_delta = float(y[i] == j)

    #         # Add (p_j(x_i) - delta) * x_i
    #         grad_sum += (ten_p[j] - kronecker_delta) * x[i]

    #     return grad_sum / N
    def gradient_L_d_W(self, x, y, W):
        N = x.shape[0]

        ten_p = self.softmax(x, W)

        # kronecker_delta = (float(y[i] == j)
        # (ten_p[j] - kronecker_delta)
        # np.arange(N) shape: (N,1), y.squeeze() shape: (N,1)
        # ten_p keeps shape: (N, 10)
        # (0, y[0]), (1, y[1]), (2, y[2])
        ten_p[np.arange(N), y.squeeze()] = ten_p[np.arange(N), y.squeeze()] - 1.0

        # x.T shape: (d, N), ten_p shape: (N, 10) to  grad_W shape: (d, 10)
        grad_W = x.T @ ten_p

        return grad_W / N

    def compute_loss_and_gradient(self):
        """
        Computes total loss and gradient of total loss with respect to weights
        w.  You may want to use the `data, w, labels, reg_wt` attributes in
        this function.

        Returns:
            data_loss, reg_loss, total_loss: 3 scalars that represent the
                losses $L_d$, $L_r$ and $L$ as defined in the README.
            grad_w: d x 10. The gradient of the total loss (including
            the regularization term), wrt the weight.
        """
        # BEGIN YOUR CODE

        x = self.data  # Shape: (N, d)
        y = self.labels  # Shape: (N, 1)
        dim = x.shape[1]
        # grad_w = np.zeros((dim, 10))
        # grad_data_loss = np.zeros((dim, 10))
        # reg_data_loss = np.zeros((dim, 10))
        # data_loss = np.zeros(10)
        # reg_loss = np.zeros(10)
        # total_loss = np.zeros(10)
        # for j in range(10):
        #     for i in range(x.shape[0]):
        #         data_loss[j] = self.L_d(x, y[i], self.w)
        #     reg_loss[j] = self.L_r(self.w[j])
        #     for d in range(dim):
        #         grad_data_loss[d][j] = self.gradient_L_d_w_j(x, y, self.w, j)
        #         reg_data_loss[d][j] = self.gradient_L_r_w_j(self.w[j])
        #         grad_w[d][j] = grad_data_loss[d][j] + reg_data_loss[d][j]
        #     total_loss[j] = reg_loss + data_loss
        #     # sigmoid(x) = 1/ (1+ exp(-w*x))
        #     # grad_w = -1 * signmoid(-yi * w * xi) * yi * xi

        # return np.sum(data_loss), np.sum(reg_loss), np.sum(total_loss), grad_w
        grad_w = np.zeros((dim, 10))  # (d, C)

        # scalars
        data_loss = self.L_d(x, y, self.w)
        reg_loss = self.L_r(self.w)
        total_loss = data_loss + reg_loss

        grad_r = self.reg_wt * self.w
        # for j in range(10):
        #     grad_d_j = self.gradient_L_d_w_j(x, y, self.w, j)  # Shape: (d,)
        # grad_w[:, j] = grad_d_j  # (d,)
        grad_d_j = self.gradient_L_d_W(x, y, self.w)  # Shape: (d,)

        grad_w = grad_r + grad_d_j

        return data_loss, reg_loss, total_loss, grad_w  # 3 scalars & (d, 10)
        # END YOUR CODE

    def train(self):
        """Train the linear classifier using gradient descent"""
        for i in range(self.epochs):
            # BEGIN YOUR CODE
            data_loss, reg_loss, total_loss, grad_w = self.compute_loss_and_gradient()
            # need to update w
            # accuracy =
            # start with some initial estimate of w, done
            # Repeat until convergence (or epochs)
            # find gradient of the loss
            # grad_w: d x 10. The gradient of the total loss

            # gradient step
            self.w = self.w - grad_w * self.lr
            print(
                f"Epoch {i+1}/{self.epochs} - Loss: {total_loss:.4f})",
                f"data_loss {data_loss:.4f} - reg_loss: {reg_loss:.4f})",
                flush=True,
            )

            # if self.writer is not None:
            #     self.writer.add_scalar(
            #         "Loss/train", total_loss, i
            #     )  # self.writer.add_scalar("Accuracy/train", accuracy, self.epochs)
            # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n_test x d matrix with a d-dimensional feature for each of the
            n_test points
        Returns:
            y: n_test vector with the predicted label for each of the n_test
            points
        """
        # self.train()
        #  X is n by d, while w is d by 10 (C=10)
        # print(x.shape)
        scores = x @ self.w
        # scores is n by 10, n rows, 10 columns for scores
        # axis = 1, means find max of each row and return index
        # axis = 0, means find max of each column and return index
        predicted_labels = np.argmax(scores, axis=1).astype(self.labels.dtype)
        # print(predicted_labels)

        # predicted_labels is n by 1 vector
        return predicted_labels


class LinearRegressionClassifier(object):
    def __init__(self, data, labels, reg_wt=3e-5):
        self.data = data
        self.labels = labels
        self.reg_wt = reg_wt
        self.w = np.zeros((self.data.shape[1], 10))

    def gradient(self, x, y_j, W, j):
        # w = (x^T * x + (reg * N * I))^-1 x^T *Y

        N = x.shape[0]
        x_T = np.transpose(x)
        x_dot = np.dot(x_T, x)
        inner_product = x_dot + self.reg_wt * N * np.identity(W.shape[0])

        return np.linalg.inv(inner_product) @ x_T @ y_j

    def compute_gradient(self):
        x = self.data  # Shape: (N, d)
        y = self.labels  # Shape: (N, 1)
        dim = x.shape[1]
        grad_w = np.zeros((dim, 10))  # (d, C)

        # print(y.shape)
        # print(y)

        for j in range(10):
            y_j = np.zeros(y.shape[0])

            for i in range(y.shape[0]):
                y_j[i] = -1.0
                if y[i] == j:
                    y_j[i] = 1.0  # if the digit out of the 100 is the class, then 1
                # else:
            # for k in range()
            grad_w[:, j] = self.gradient(x, y_j, self.w, j)  # (d,)

        return grad_w  # (d, 10)

    def train(self):
        """Train the linear classifier using closed form solution"""
        n_train, d = self.data.shape
        # BEGIN YOUR CODE
        grad_w = self.compute_gradient()
        self.w = grad_w

        # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n_test x d matrix with a d-dimensional feature for each of the
            n_test points
        Returns:
            y: n_test vector with the predicted label for each of the n_test
            points
        """
        # BEGIN YOUR CODE
        self.train()
        print(x.shape)
        scores = x @ self.w
        # print(scores.shape)
        # print(scores)
        predicted_labels = np.argmax(scores, axis=1).astype(self.labels.dtype)
        # print(predicted_labels)

        return predicted_labels
        # END YOUR CODE
