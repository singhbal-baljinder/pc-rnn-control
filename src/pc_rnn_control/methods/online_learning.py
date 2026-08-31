"""
Unified Online Learning Model Class

This module provides a single interface for using different online learning models.
All model initialization is handled internally - users just specify the model type
and basic parameters (input_dim, output_dim, hidden_size, etc.).
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional, Any

# Temporal PC imports
from pc_rnn_control.methods.predictive_coding import MyTemporalPC

# LWPR imports. pylwpr is a hand-built C++ extension (LGPL, Edinburgh); it is
# not on PyPI, so it is an optional prerequisite. Missing it must disable the
# LWPR baseline only -- the `if LWPR is None` guard below handles that -- and
# not take every other method down with it.
try:
    from pylwpr import LWPR
except ImportError:  # pragma: no cover - depends on the local build
    try:
        from lwpr import LWPR
    except ImportError:
        LWPR = None

# LogGP imports
from pc_rnn_control.methods.gp.log_gp import LoGGP

# Online RNN imports
from pc_rnn_control.methods.rnn.network import RNN
from pc_rnn_control.methods.rnn.learning_algorithms import (
    RTRL,
    KF_RTRL,
    UORO,
    Reverse_KF_RTRL,
    RFLO,
    DNI,
    Efficient_BPTT,
    Future_BPTT,
    KeRNL,
)
from pc_rnn_control.methods.rnn.functions import tanh, sigmoid, identity, mean_squared_error
from pc_rnn_control.methods.rnn.optimizers import Stochastic_Gradient_Descent
from pc_rnn_control.methods.rnn.simulation import Simulation

# CerebNet imports
try:
    from pc_rnn_control.methods.cerebnet import CerebNet
except ImportError:
    CerebNet = None


class OnlineLearningModel:
    """
    A unified wrapper for online learning models with a consistent API.

    Supported models:
    - 'temporal_pc': Temporal Predictive Coding
    - 'lwpr': Locally Weighted Projection Regression
    - 'log_gp': Logarithmic Gaussian Process
    - 'cerebnet': Cerebellum-inspired algorithm

    Usage:
    ------
    model = OnlineLearningModel("lwpr", input_dim=10, output_dim=9)
    pred = model.predict(x)
    model.update(x, y)
    """

    def __init__(
        self, model_type: str, input_dim: int = None, output_dim: int = None, **kwargs
    ):
        """
        Initialize the OnlineLearningModel.

        Parameters
        ----------
        model_type : str
            Type of model. Options: 'temporal_pc', 'lwpr', 'log_gp',
            'cerebnet', 'rnn-<algorithm>'
        input_dim : int
            Input dimension (required for all models)
        output_dim : int
            Output dimension (required for all models)
        **kwargs : dict
            Optional model-specific parameters:
            - hidden_size: hidden layer size for RNN/Temporal PC/CerebNet (default: 50)
            - max_stored_data: max samples for LogGP (default: 100)
            - num_local_models: number of local models for LogGP (default: 100)
            - learning_rate: learning rate for RNN (default: 0.001)
            - alpha: inference learning rate for Temporal PC (default: 1e-3)
            - alpha_learning: learning learning rate for Temporal PC (default: 1e-3)
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None

        if input_dim is None or output_dim is None:
            raise ValueError("input_dim and output_dim are required for all models")

        # Make sure all corresponding parameters are initiazed with the same values
        self.hidden_size = kwargs.get("hidden_size", 100)

        self.W_in_0 = kwargs.get(
            "W_in",
            np.random.normal(
                0, np.sqrt(1 / self.input_dim), (self.hidden_size, self.input_dim)
            ),
        )
        self.W_rec_0 = kwargs.get(
            "W_rec",
            np.random.normal(
                0, np.sqrt(1 / self.hidden_size), (self.hidden_size, self.hidden_size)
            ),
        )

        self.W_out_0 = kwargs.get(
            "W_out",
            np.random.normal(
                0, np.sqrt(1 / self.hidden_size), (self.output_dim, self.hidden_size)
            ),
        )

        self.b_rec_0 = kwargs.get("b_rec", np.zeros(self.hidden_size))

        self.b_out_0 = kwargs.get("b_out", np.zeros(self.output_dim))

        self.x_0 = kwargs.get(
            "x_0",
            np.random.normal(0, np.sqrt(1 / self.hidden_size), (self.hidden_size,)),
        )

        self._init_model(model_type, **kwargs)

    def _init_model(self, model_type: str, **kwargs):
        """Initialize the underlying model based on type."""
        if model_type.startswith("temporal_pc"):
            self._init_temporal_pc(**kwargs)
        elif model_type == "lwpr":
            self._init_lwpr(**kwargs)
        elif model_type == "log_gp":
            self._init_log_gp(**kwargs)
        elif model_type == "cerebnet":
            self._init_cerebnet(**kwargs)
        elif model_type.startswith("rnn-"):
            # Extract algorithm name from model_type like "rnn-uoro"
            alg_name = model_type.replace("rnn-", "")
            self._init_online_rnn(alg_name, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _init_temporal_pc(self, **kwargs):
        """Initialize Temporal Predictive Coding model."""
        if MyTemporalPC is None:
            raise ImportError(
                "MyTemporalPC not available. Install predictive_dyn_models module."
            )

        hidden_size = kwargs.get("hidden_size", 100)
        alpha = kwargs.get("alpha", 1e-3)
        alpha_learning = kwargs.get("alpha_learning", 1e-3)
        num_inference_iters = kwargs.get("num_inference_iters", 5)
        num_learning_iters = kwargs.get("num_learning_iters", 5)
        learn_Wr = kwargs.get("learn_Wr", True)
        learn_Win = kwargs.get("learn_Win", True)
        learn_Wout = kwargs.get("learn_Wout", True)
        use_output_bias = kwargs.get("use_output_bias", True)
        use_input_bias = kwargs.get("use_input_bias", True)
        infer_precision = kwargs.get("infer_precision", True)
        use_known_input = kwargs.get("use_known_input", True)
        Sigma_x_init = kwargs.get("Sigma_x_init", 1e2)
        Sigma_y_init = kwargs.get("Sigma_y_init", 1e2)
        activation_fun = kwargs.get("activation_fun", "tanh")
        # Initialize Temporal PC model
        init_state = MyTemporalPC.init_hidden(hidden_size, bsz=1)
        self.model = MyTemporalPC(
            control_size=self.input_dim,
            hidden_size=hidden_size,
            output_size=self.output_dim,
            init_hidden_state=torch.from_numpy(self.x_0)
            .float()
            .reshape(1, self.hidden_size),
            nonlin=activation_fun,
            learn_Win=learn_Win,
            learn_Wr=learn_Wr,
            learn_Wout=learn_Wout,
            use_output_bias=use_output_bias,
            use_input_bias=use_input_bias,
            infer_precision=infer_precision,
            use_known_input=use_known_input,
            Sigma_x_init=Sigma_x_init,
            Sigma_y_init=Sigma_y_init,
            Win_0=torch.tensor(self.W_in_0),
            Wrec_0=torch.tensor(self.W_rec_0),
            Wout_0=torch.tensor(self.W_out_0),
            bout_0=torch.tensor(self.b_out_0),
            brec_0=torch.tensor(self.b_rec_0),
        )

        # Store hyperparameters
        self.alpha = alpha
        self.alpha_learning = alpha_learning
        self.num_inference_iters = num_inference_iters
        self.num_learning_iters = num_learning_iters
        self.prev_state = init_state
        self.prev_precision_x = torch.ones(1, hidden_size)
        self.prev_precision_y = torch.ones(1, self.output_dim)
        self.idx_iter = 0



    def _init_lwpr(self, **kwargs):
        """Initialize LWPR model."""
        if LWPR is None:
            raise ImportError("LWPR not available. Install lwpr module.")

        # init_D / init_alpha are the initial distance metric and the initial
        # 2nd-order learning rate -- LWPR has no RNG of its own, so these are the
        # only free initialisation it has. Exposing them lets a caller sweep them
        # the way a stochastic method is swept over random seeds. The defaults
        # are the values previously hardcoded here, so callers that pass nothing
        # are unaffected.
        init_D = float(kwargs.get("init_D", 40.0))
        init_alpha = float(kwargs.get("init_alpha", 250.0))

        self.model = LWPR(self.input_dim, self.output_dim)
        self.model.init_D = init_D * np.eye(self.input_dim)
        self.model.init_alpha = init_alpha * np.ones([self.input_dim, self.input_dim])
        self.model.meta = True  # Use meta-learning
        self.model.update_D = True  # Update distance metric

    def _init_log_gp(self, **kwargs):
        """Initialize LogGP model."""
        if LoGGP is None:
            raise ImportError("LoGGP not available. Install log_gp module.")

        max_stored_data = kwargs.get("max_stored_data", 100)
        num_local_models = kwargs.get("num_local_models", 100)

        # LoGGP(xSize, outs, pts, N, ard)
        # xSize = input_dim, outs = output_dim, pts = max_stored_data, N = num_local_models, ard = True
        self.model = LoGGP(
            xSize=self.input_dim,
            outs=self.output_dim,
            pts=max_stored_data,
            N=num_local_models,
            ard=True,
        )


    def _init_cerebnet(self, **kwargs):
        """Initialize Cerebellum-inspired model."""
        if CerebNet is None:
            raise ImportError("CerebNet not available.")

        hidden_size = kwargs.get("hidden_size", 100)
        tau = kwargs.get("tau", 0.01)
        eta = kwargs.get("eta", 1e-3)
        lambda_reg = kwargs.get("lambda_reg", 0.02)
        self.num_iters = kwargs.get("num_iters", 1)

        self.model = CerebNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_size=hidden_size,
            tau=tau,
            eta=eta,
            lambda_reg=lambda_reg,
            Win_0=self.W_in_0,
            Wrec_0=self.W_rec_0,
            Wout_0=self.W_out_0,
            brec_0=self.b_rec_0,
            z_0=self.x_0,
        )

    def _init_online_rnn(self, algorithm: str = "rtrl", **kwargs):
        """Initialize Online RNN model with specified learning algorithm."""
        if RNN is None or RTRL is None:
            raise ImportError("Online RNN not available. Install online_rnn module.")

        hidden_size = kwargs.get("hidden_size", 100)
        learning_rate = kwargs.get("learning_rate", 1e-3)
        truncation = kwargs.get("truncation", 100)
        self.num_iters = kwargs.get("num_iters", 1)
        activation = kwargs.get("activation_function", "sigmoid")
        if activation == "sigmoid":
            act_fun = sigmoid
        elif activation == "tanh":
            act_fun = tanh
        elif activation == "identity":
            act_fun = identity
        else:
            raise RuntimeError(
                "Unsupported activation function for RNN based methods " + activation
            )
        # Create RNN
        rnn = RNN(
            self.W_in_0,
            self.W_rec_0,
            self.W_out_0,
            self.b_rec_0,
            self.b_out_0,
            h=self.x_0,
            activation=act_fun,
            alpha=1,
            output=identity,
            loss=mean_squared_error,
        )

        # Create learning algorithm based on algorithm name
        self.optimizer = Stochastic_Gradient_Descent(lr=learning_rate)
        algorithm = algorithm.lower()
        if algorithm == "rtrl":
            self.learning_algorithm = RTRL(rnn)
        elif algorithm == "kf_rtrl" or algorithm == "kf-rtrl":
            self.learning_algorithm = KF_RTRL(rnn)
        elif algorithm == "uoro":
            self.learning_algorithm = UORO(rnn)
        elif algorithm == "reverse_kf_rtrl" or algorithm == "reverse-kf-rtrl":
            self.learning_algorithm = Reverse_KF_RTRL(rnn)
        elif algorithm == "rflo":
            self.learning_algorithm = RFLO(rnn, alpha=1)
        elif algorithm == "dni":
            self.learning_algorithm = DNI(
                rnn, optimizer=Stochastic_Gradient_Descent(lr=learning_rate)
            )
        elif algorithm == "efficient_bptt" or algorithm == "efficient-bptt":
            self.learning_algorithm = Efficient_BPTT(rnn, T_truncation=truncation)
        elif algorithm == "future_bptt" or algorithm == "future-bptt":
            self.learning_algorithm = Future_BPTT(rnn, T_truncation=truncation)
        elif algorithm == "kernl" or algorithm == "kernl":
            self.learning_algorithm = KeRNL(
                rnn, optimizer=Stochastic_Gradient_Descent(lr=learning_rate)
            )
        else:
            # Default to RTRL
            raise ValueError(f"Unknown algorithm type: {algorithm}")

        self.rnn = rnn
        self.rnn_algorithm = algorithm

        # Creata simulation object (contains wrappers to perform training step cleanly)
        self.simulation_rnn = Simulation(self.rnn)

        # To avoid errors, initialize "previous"
        # inputs/labels as the first inputs/labels
        self.rnn.x_prev = np.zeros((self.input_dim,))
        self.rnn.y_prev = np.zeros((self.output_dim,))

    def predict(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make a prediction given input x.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input to the model. Shape depends on model type.

        Returns
        -------
        prediction : np.ndarray
            Model prediction.
        """
        if self.model_type.startswith("temporal_pc"):
            return self._predict_temporal_pc(x)
        elif self.model_type == "lwpr":
            return self._predict_lwpr(x)
        elif self.model_type == "log_gp":
            return self._predict_log_gp(x)
        elif self.model_type == "cerebnet":
            return self._predict_cerebnet(x)
        elif self.model_type.startswith("rnn-"):
            return self._predict_online_rnn(x)

    def _predict_temporal_pc(
        self,
        u: np.ndarray,
    ) -> np.ndarray:
        """Predict with Temporal PC."""
        # Convert to tensor if needed
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u).float()

        # Ensure proper shape
        if u.dim() == 1:
            u = u.unsqueeze(0)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        # Forward pass
        pred_y, _ = self.model.forward(u=u)

        # Store for later use in update
        self._last_prediction = pred_y.detach()

        # Return as numpy arrays
        return pred_y.detach().numpy().flatten()

    def _predict_lwpr(self, x: np.ndarray) -> np.ndarray:
        """Predict with LWPR."""
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()

        prediction, confidence = self.model.predict_conf(x.flatten())
        self._last_confidence = confidence

        return prediction

    def _predict_log_gp(self, x: np.ndarray) -> np.ndarray:
        """Predict with LogGP."""
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()

        # LogGP expects input shape (input_dim,) and returns scalar per output dim
        prediction = self.model.predict(x.flatten())

        return prediction


    def _predict_cerebnet(self, x: np.ndarray) -> np.ndarray:
        """Predict with CerebNet."""
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()

        x_flat = x.flatten()
        y_pred = self.model.predict(x_flat)

        # Store for computing error during update
        self._last_cerebnet_pred = y_pred
        return y_pred

    def _predict_online_rnn(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Online RNN prediction - single sample forward pass."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # Ensure x is 1D
        x_flat = np.squeeze(x)
        if x_flat.ndim == 0:
            x_flat = x_flat.reshape(-1)

        # Forward pass through RNN
        self.rnn.next_state(x_flat)
        self.rnn.z_out()

        # Get output
        try:
            output = self.rnn.output.f(self.rnn.z)
            if isinstance(output, torch.Tensor):
                return output.detach().cpu().numpy().flatten()
            else:
                return np.array(output).flatten()
        except Exception:
            # Fallback if output function fails
            return np.zeros(self.output_dim)

    def update(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        prior=None,
    ):
        """
        Update the model with new data.

        Parameters
        ----------
        x : np.ndarray
            Input data
        y : np.ndarray
            Target output data
        """
        if self.model_type.startswith("temporal_pc"):
            self._update_temporal_pc(x, y, prior)
        elif self.model_type == "lwpr":
            self._update_lwpr(x, y)
        elif self.model_type == "log_gp":
            self._update_log_gp(x, y)
        elif self.model_type == "cerebnet":
            self._update_cerebnet(x, y)
        elif self.model_type.startswith("rnn-"):
            self._update_online_rnn(x, y)

    def _update_temporal_pc(self, u: np.ndarray, y: np.ndarray, prior=None):
        """Update Temporal PC with inference and learning steps."""
        # Convert to tensors if needed
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # Ensure proper shapes
        if u.dim() == 1:
            u = u.unsqueeze(0)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        # Inference step - infer latent states
        self.model.infer(
            inf_iters=self.num_inference_iters,
            y=y,
            u=u,
            alpha_inf=self.alpha,
            alpha_lr=self.alpha_learning,
            mode="infer_states",
            prior=prior,
        )

        # Learning step - update parameters
        self.model.infer(
            inf_iters=self.num_learning_iters,
            y=y,
            u=u,
            alpha_inf=self.alpha,
            alpha_lr=self.alpha_learning,
            mode="learn_param",
            prior=prior,
        )

        self.idx_iter = self.idx_iter + 1

    def _update_lwpr(self, x: np.ndarray, y: np.ndarray):
        """Update LWPR model."""
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()

        self.model.update(x.flatten(), y.flatten())

    def _update_log_gp(self, x: np.ndarray, y: np.ndarray):
        """Update LogGP model."""
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()

        self.model.update(x.flatten(), y.flatten())


    def _update_cerebnet(self, x: np.ndarray, y: np.ndarray):
        """Update CerebNet model."""
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()
        y_flat = y.flatten()

        # CerebNet uses the raw error: y_target - y_pred
        error = y_flat - self._last_cerebnet_pred
        for _ in range(self.num_iters):
            self.model.update(error)

    def _update_online_rnn(
        self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ):
        """Online RNN learning."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # Flatten inputs for RNN
        x_flat = np.squeeze(x)
        if x_flat.ndim == 0:
            x_flat = x_flat.reshape(-1)
        y_flat = np.squeeze(y)
        if y_flat.ndim == 0:
            y_flat = y_flat.reshape(-1)

        # Use method from Simulation class to perform training
        # Having passed ther rnn as a reference it will update it
        # for the predict step
        # NOTE: predict() already advanced the network one step on this same
        # x_flat (next_state + z_out). We must NOT advance it again here, or the
        # state is stepped twice per sample and the gradient is computed on a
        # different forward pass than the one that was scored. Reuse the
        # prediction forward pass and only attach the label/error.
        self.rnn.x = x_flat
        self.rnn.y = y_flat

        for _ in range(self.num_iters):

            # Reuse the forward pass produced by predict() (self.rnn.z, a, h,
            # a_prev are all already set); just compute the immediate error.
            self.rnn.y_hat = self.rnn.output.f(self.rnn.z)

            self.rnn.error = self.rnn.loss.f_prime(self.rnn.z, self.rnn.y)

            # Update learn_alg variables and get gradients
            self.learning_algorithm.update_learning_vars()
            self.grads_list = self.learning_algorithm()

            ### --- Pass gradients to optimizer --- ###
            # Get updated parameters
            self.rnn.params = self.optimizer.get_updated_params(
                self.rnn.params, self.grads_list
            )

            (
                self.rnn.W_rec,
                self.rnn.W_in,
                self.rnn.b_rec,
                self.rnn.W_out,
                self.rnn.b_out,
            ) = self.rnn.params

        # Current inputs/labels become previous inputs/labels
        self.rnn.x_prev = self.rnn.x.copy()
        self.rnn.y_prev = self.rnn.y.copy()

    def predict_and_update(
        self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict then update the model in one call.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data
        y : np.ndarray or torch.Tensor
            Target output data

        Returns
        -------
        prediction : np.ndarray or Tuple[np.ndarray, np.ndarray]
            Model prediction before update
        """
        prediction = self.predict(x)
        self.update(x, y)
        return prediction

    def get_model(self) -> Any:
        """Get the underlying model object."""
        return self.model

    def get_model_type(self) -> str:
        """Get the model type string."""
        return self.model_type
