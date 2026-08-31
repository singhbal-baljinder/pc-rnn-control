import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class CerebNet:
    """
    Cerebellum-inspired adaptive kinematic controller
    (Kalidindi et al., 2019)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=300,
        tau=0.01,
        eta=0.001,
        lambda_reg=0.02,
        Win_0=None,
        Wrec_0=None,
        brec_0=None,
        Wout_0=None,
        z_0=None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.tau = tau
        self.eta = (
            eta  # Note: 8.0 in paper, defaulted lower for normalized data stability
        )
        self.lambda_reg = lambda_reg

        # If not provided by the user, initialize from paper Section III-B
        # "both w_in and w are generated from random distributions"
        # "w are assigned strictly positive values"
        if Win_0 is None:
            self.W_in = np.random.randn(hidden_size, input_dim) * 0.1
        else:
            self.W_in = Win_0

        if Wrec_0 is None:
            self.W_rec = np.abs(np.random.randn(hidden_size, hidden_size)) * 0.1
        else:
            self.W_rec = np.abs(Wrec_0) * 0.1

        if brec_0 is None:
            self.b = np.random.randn(hidden_size) * 0.1
        else:
            self.b = brec_0

        # CerebNet seems unstable for random intial Wout weights
        if Wout_0 is None:
            self.W_out = np.abs(np.random.randn(output_dim, hidden_size)) * 0.1
        else:
            self.W_out = Wout_0 * 0.1

        if z_0 is None:
            self.z = np.zeros(hidden_size)
        else:
            self.z = z_0

    def predict(self, u):
        # Eq 1: tau z(t) = [-z(t) + f(w_in u(t) - w z(t)) + b]^+
        # Implemented as a discrete Euler step for simulation stability
        inner = np.dot(self.W_in, u) - np.dot(self.W_rec, self.z)
        dz = -self.z + sigmoid(inner) + self.b
        self.z = np.maximum(0, self.z + self.tau * dz)

        # Eq 2: y_out(t) = -w_out p(t)   (where p(t) = z(t) as per paper)
        y_out = -np.dot(self.W_out, self.z)
        return y_out

    def update(self, error):
        # Eq 3: delta_w_out = -eta * p * sign(e) + lambda * w_out^3
        sign_e = np.sign(error)

        # Outer product to compute weight updates for the entire matrix
        delta_W_out = -self.eta * np.outer(sign_e, self.z) + self.lambda_reg * (
            self.W_out**3
        )
        self.W_out += delta_W_out
