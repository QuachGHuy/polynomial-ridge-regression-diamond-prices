from typing import Optional
import numpy as np

class PolynomialRidge:
    """
    Polynomial Ridge Regression using mini-batch gradient descent.
    """

    def __init__(self, alpha: float = 0.0):
        """
        Parameters
        ----------
        alpha : float
            Ridge regularization strength.
        """
        self.alpha = alpha
        self.theta: Optional[np.ndarray] = None
        self.loss_history: list[float] = []

    # --------------------
    # Core methods
    # --------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int,
        random_state: Optional[int] = None,
    ) -> "PolynomialRidge":
        """
        Train the model using mini-batch gradient descent.
        """

        if random_state is not None:
            np.random.seed(random_state)

        n_samples, n_features = X.shape
        self.theta = np.random.randn(n_features)
        self.loss_history.clear()

        for _ in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            batch_losses = []

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.predict(X_batch)

                loss = self._ridge_mse(y_batch, y_pred)
                batch_losses.append(loss)

                gradient = self._gradient(X_batch, y_batch, y_pred)
                self.theta -= lr * gradient

            self.loss_history.append(float(np.mean(batch_losses)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.
        """

        if self.theta is None:
            raise RuntimeError("Model has not been fitted yet.")

        if X.shape[1] != self.theta.shape[0]:
            raise ValueError("Input feature dimension mismatch.")

        return X @ self.theta

    # --------------------
    # Loss & gradient
    # --------------------

    def _ridge_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_pred - y_true

        theta_ridge = self.theta.copy()
        theta_ridge[0] = 0  # do not regularize bias

        mse = np.mean(error ** 2)
        ridge_penalty = self.alpha * np.sum(theta_ridge ** 2)

        return mse + ridge_penalty

    def _gradient(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray: 
        n = X.shape[0]
        error = y_pred - y_true

        grad_mse = (2 / n) * (X.T @ error)

        theta_ridge = self.theta.copy()
        theta_ridge[0] = 0

        grad_ridge = 2 * self.alpha * theta_ridge

        return grad_mse + grad_ridge

    # --------------------
    # Persistence
    # --------------------

    def save(self, path: str) -> None:
        """
        Save model weights.
        """
        if self.theta is None:
            raise RuntimeError("Nothing to save. Model not fitted.")

        np.save(path, self.theta)

    def load(self, path: str) -> None:
        """
        Load model weights.
        """
        self.theta = np.load(path)

