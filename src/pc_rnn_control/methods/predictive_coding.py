import torch
import torch.nn as nn

import numpy as np


VERY_SMALL_NUMBER = 1e-6
VERY_LARGE_NUMBER = 1e6
# Inference-time clamps on the running posterior precisions Sigma_x / Sigma_y.
# Prevent numerical blowup during gradient descent in infer().
MAX_PRECISION = 10
MIN_PRECISION = 1e-3



class Tanh(nn.Module):
    def forward(self, inp):
        if inp is not None:
            return torch.tanh(inp)
        else:
            raise ValueError("Input cannot be None")

    def deriv(self, inp):
        if inp is not None:
            return 1.0 - torch.tanh(inp) ** 2.0
        else:
            raise ValueError("Input cannot be None")


class Linear(nn.Module):
    def forward(self, inp):
        if inp is not None:
            return inp
        else:
            raise ValueError("Input cannot be None")

    def deriv(self, inp):
        if inp is not None:
            return torch.ones_like(inp).to(inp.device)
        else:
            raise ValueError("Input cannot be None")


class Sigmoid(nn.Module):
    def forward(self, inp):
        if inp is not None:
            return torch.sigmoid(inp)
        else:
            raise ValueError("Input cannot be None")

    def deriv(self, inp):
        if inp is not None:
            s = torch.sigmoid(inp)
            return s * (1.0 - s)
        else:
            raise ValueError("Input cannot be None")




class MyTemporalPC(nn.Module):
    def __init__(
        self,
        control_size,
        hidden_size,
        output_size,
        learn_Wr=True,
        learn_Win=True,
        learn_Wout=True,
        nonlin="tanh",
        init_hidden_state=None,
        use_input_bias=False,
        use_output_bias=False,
        infer_precision=False,
        use_known_input=True,
        save_weight_history=True,
        Sigma_x_init=1e1,
        Sigma_y_init=1e1,
        Win_0=None,
        Wout_0=None,
        Wrec_0=None,
        brec_0=None,
        bout_0=None,
        x0=None,
    ):
        """Model inspired by Millidge's Temporal Predictive Coding (TPC) model.
        x_k+1 = Wr * f(x_k) + Win * u_k + gaussian_noise
        y_k = Wout * f(x_k) + gaussian_noise
        Note: both observation and state transition models use the same f function, but different weigth matrices.
        Args:
            control_size (int): Size of the control input vector u_k.
            hidden_size (int): Size of the hidden state vector x_k.
            output_size (int): Size of the output vector y_k.
            nonlin (str): Nonlinearity to use ('linear' or 'tanh').
        """
        super(MyTemporalPC, self).__init__()
        self.hidden_size = hidden_size
        self.control_size = control_size
        self.output_size = output_size
        self.use_input_bias = use_input_bias
        self.use_output_bias = use_output_bias
        self.learn_Win = learn_Win
        # Allow user to use known input u and its weghts Win
        self.use_known_input = use_known_input
        self.learn_Wout = learn_Wout
        self.learn_Wr = learn_Wr
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=self.use_input_bias)
        if Wrec_0 is None:
            self.Wr.weight.data.copy_(
                torch.linalg.qr(torch.randn(hidden_size, hidden_size))[0]
            )
        else:
            self.Wr.weight.data.copy_(Wrec_0)
        # self.Wr.weight.data.copy_(
        #    torch.zeros(hidden_size, hidden_size)
        # )
        # If bias is used, initialize to zero
        if self.use_input_bias:
            if brec_0 is None:
                self.Wr.bias.data.copy_(torch.zeros(hidden_size))
            else:
                self.Wr.bias.data.copy_(brec_0)
        self.infer_precision = infer_precision

        self.Wout = nn.Linear(hidden_size, output_size, bias=self.use_output_bias)
        if self.use_output_bias:
            if bout_0 is None:
                self.Wout.bias.data.copy_(torch.zeros(output_size))
            else:
                self.Wout.bias.data.copy_(bout_0)

        if Wout_0 is None:
            self.Wout.weight.data.copy_(
                torch.normal(
                    mean=0,
                    std=1 / np.sqrt(hidden_size),
                    size=(output_size, hidden_size),
                )
            )
        else:
            self.Wout.weight.data.copy_(Wout_0)

        # with torch.no_grad():
        #    s = min(self.output_size, self.hidden_size)
        #    self.Wout.weight[:s, :s] = torch.eye(s)/s

        if self.infer_precision:
            self.Sigma_x = torch.ones((1, hidden_size)) * Sigma_x_init
            self.Sigma_y = torch.ones((1, output_size)) * Sigma_y_init
        else:
            self.Sigma_x = torch.ones((1, hidden_size)) * Sigma_x_init
            self.Sigma_y = torch.ones((1, output_size)) * Sigma_y_init

        self.hidden_loss = None
        self.obs_loss = None
        self.x = init_hidden_state
        self.prev_pred_x = self.x
        self.Win = nn.Linear(control_size, hidden_size, bias=False)
        if Win_0 is None:
            self.Win.weight.data.copy_(
                torch.normal(
                    mean=0,
                    std=1 / np.sqrt(control_size),
                    size=(hidden_size, control_size),
                )
            )
        else:
            self.Win.weight.data.copy_(Win_0)

        # Set the top-left control_size x control_size block of Win to identity
        # with torch.no_grad():
        #    s = min(self.control_size, self.hidden_size)
        #    self.Win.weight[:s, :s] = torch.eye(s)/s

        self.x_history = []
        if nonlin == "linear":
            self.nonlin = Linear()
        elif nonlin == "tanh":
            self.nonlin = Tanh()
        elif nonlin == "sigmoid":
            self.nonlin = Sigmoid()
        else:
            raise ValueError("no such nonlinearity!")

        self.save_weight_history = save_weight_history
        self.delta_x = []
        self.delta_Sigma_x = []
        self.delta_Sigma_y = []
        self.delta_Wr = []
        self.delta_Win = []
        self.delta_Wout = []
        self.delta_b = []
        self.delta_bout = []
        # Save initial weights
        self.Wr_init = self.Wr.weight.data.clone()
        self.Win_init = self.Win.weight.data.clone()
        self.Wout_init = self.Wout.weight.data.clone()
        if self.use_input_bias:
            self.b_init = self.Wr.bias.data.clone()

        # Save weights history
        if self.save_weight_history:
            self.Wr_history = [self.Wr.weight.data.clone()]
            self.Win_history = [self.Win.weight.data.clone()]
            self.Wout_history = [self.Wout.weight.data.clone()]
            if self.use_input_bias:
                self.b_history = [self.Wr.bias.data.clone()]
            if self.use_output_bias:
                self.bout_history = [self.Wout.bias.data.clone()]
            self.x_history = []




    def forward(self, u):

        batch_size, seq_len, _ = u.size()
        pred_y = torch.zeros(batch_size, seq_len, self.output_size)
        pred_x = self.x.clone().detach()
        for i in range(0, seq_len):
            # Emit y_k from the current state first (paper Eq. 5:
            # y_k = W_out f(x_{k-1})), then transition. Using self.x here
            # (the bug in the previous version) made every pred_y[i] identical
            # for seq_len > 1, silently breaking multi-step prediction.
            pred_y[:, i, :] = self.Wout(self.nonlin(pred_x))
            # Recurrent transition: x_{k} = W_r f(x_{k-1}) + W_in f(u_{k-1}).
            if self.use_known_input:
                pred_x = self.Wr(self.nonlin(pred_x)) + self.Win(
                    self.nonlin(u[:, i, :])
                )
            else:
                pred_x = self.Wr(self.nonlin(pred_x))

        return pred_y, pred_x

    @staticmethod
    def init_hidden(hidden_size, bsz):
        """Initializing prev_x"""
        return torch.randn(1, hidden_size)

    def infer(
        self,
        inf_iters,
        y,
        u,
        alpha_inf=1e-5,
        alpha_lr=1e-5,
        mode: str = "infer_states",
        log=True,
        prior=None,
    ):
        """
        Hidden state inference using gradient descent on free energy
        This function implements the discrete version of:
            x_dot = -error_x + f'(x) O Wout^T*error_y
        Where O is the element wise multiplication
        """

        with torch.no_grad():
            tot_grad_vfe_x = 0
            tot_grad_vfe_sigma_x = 0
            tot_grad_vfe_sigma_y = 0
            tot_grad_vfe_Wout = 0
            tot_grad_vfe_Win = 0
            tot_grad_vfe_Wr = 0
            tot_grad_vde_b = 0
            tot_grad_vde_bout = 0
            batch_size, seq_len, _ = u.size()
            y = y[-1, :, :]
            if mode == "infer_states":

                for i in range(inf_iters):

                    pred_y, pred_x = self.forward(u)
                    pred_y = pred_y[-1, :, :]

                    err_y_pre_sigma = y - pred_y
                    err_y = self.Sigma_y * err_y_pre_sigma

                    # State Inference as ONE gradient descent step on free energy wrt the hidden state
                    activation_deriv = self.nonlin.deriv(self.x)

                    # Transfor activation_deriv into a diagonal matrix
                    activation_deriv = torch.diag_embed(activation_deriv.squeeze(0))

                    # Deep models allow top-down prior information
                    if prior is None:
                        # If prior is provided, use it as the initial state
                        x_next_prior = self.x
                    else:
                        x_next_prior = prior
                    # State error is given by the prior on the higher order
                    # derivative (x_dot_prior) minus the prediction with the previous posterior(pred_x)
                    # Since we do not have hierarchical priors and our highest order
                    #  derivative is the first one, we can set x_dot_prior = x_posterior_prev
                    err_x_pre_sigma = x_next_prior - pred_x
                    err_x_after_sigma = self.Sigma_x * err_x_pre_sigma

                    grad_vfe_x_1 = err_y @ (-activation_deriv @ self.Wout.weight.T).T
                    grad_vfe_x_2 = (
                        err_x_after_sigma @ (-activation_deriv @ self.Wr.weight.T).T
                    )
                    grad_vfe_x = grad_vfe_x_1 + grad_vfe_x_2
                    # Compute the delta for the hidden state,
                    # If batch version is used, compute the mean gradient
                    if len(grad_vfe_x.size()) == 3:
                        grad_vfe_x = torch.mean(grad_vfe_x, dim=0)
                    # Log the same slice that is actually applied below, so the
                    # delta_x history matches the in-place update.
                    tot_grad_vfe_x += -alpha_inf * grad_vfe_x[-1, :]


                    self.x += -alpha_inf * (grad_vfe_x)[-1, :]

                    if self.infer_precision:
                        # Make sure the precision is not too low or too high
                        self.Sigma_x = torch.clamp(
                            self.Sigma_x, min=MIN_PRECISION, max=MAX_PRECISION
                        )

                        # Make sure the precision is not too low or too high
                        self.Sigma_y = torch.clamp(
                            self.Sigma_y, min=MIN_PRECISION, max=MAX_PRECISION
                        )

                        # Update variational free energy gradients wrt variances (independence assumption)
                        dVFE_dSigma_x = ((err_x_pre_sigma) ** 2) - 0.5 * (
                            1 / (self.Sigma_x)
                        )
                        dVFE_dSigma_y = 0.5 * (
                            ((err_y_pre_sigma) ** 2) - (1 / (self.Sigma_y))
                        )
                        # If batch version is used, compute the mean gradient
                        if len(dVFE_dSigma_x.size()) == 3:
                            dVFE_dSigma_x = torch.mean(dVFE_dSigma_x, dim=0)
                        # If batch version is used, compute the mean gradient
                        if len(dVFE_dSigma_y.size()) == 3:
                            dVFE_dSigma_y = torch.mean(dVFE_dSigma_y, dim=0)
                        # Explicit time integration step (negative because it is a gradient DESCENT step)
                        self.Sigma_x += -alpha_inf * dVFE_dSigma_x
                        self.Sigma_y += -alpha_inf * dVFE_dSigma_y

                        # Make sure the precision is not too low or too high
                        self.Sigma_x = torch.clamp(
                            self.Sigma_x, min=MIN_PRECISION, max=MAX_PRECISION
                        )

                        # Make sure the precision is not too low or too high
                        self.Sigma_y = torch.clamp(
                            self.Sigma_y, min=MIN_PRECISION, max=MAX_PRECISION
                        )

                        tot_grad_vfe_sigma_x = -alpha_inf * dVFE_dSigma_x
                        tot_grad_vfe_sigma_y = -alpha_inf * dVFE_dSigma_y

                if self.save_weight_history:
                    self.delta_x.append(tot_grad_vfe_x)

                if self.infer_precision and self.save_weight_history:
                    self.delta_Sigma_x.append(tot_grad_vfe_sigma_x)
                    self.delta_Sigma_y.append(tot_grad_vfe_sigma_y)

            elif mode == "learn_param":

                for i in range(inf_iters):

                    pred_y, pred_x = self.forward(u)
                    pred_y = pred_y[-1, :, :]
                    # Deep models allow top-down prior information
                    if prior is None:
                        # If prior is provided, use it as the initial state
                        x_next_prior = self.x
                    else:
                        x_next_prior = prior
                    err_x_pre_sigma = x_next_prior - pred_x
                    err_x_after_sigma = self.Sigma_x * err_x_pre_sigma
                    err_y_pre_sigma = y - pred_y
                    err_y = self.Sigma_y * err_y_pre_sigma

                    # Update weight matrices (explicit integration of a gradient descent step)
                    if self.learn_Wr:
                        grad_vfe_Wr = err_x_after_sigma.T @ (-self.nonlin(self.x))
                        # If batch version is used, compute the mean gradient
                        if len(grad_vfe_Wr.size()) == 3:
                            grad_vfe_Wr = torch.mean(grad_vfe_Wr, dim=0)
                        self.Wr.weight += -alpha_lr * grad_vfe_Wr
                        tot_grad_vfe_Wr += -alpha_lr * grad_vfe_Wr

                    if self.learn_Wout:
                        grad_vfe_Wout = err_y.mT @ (-self.nonlin(self.x))
                        # If batch version is used, compute the mean gradient
                        if len(grad_vfe_Wout.size()) == 3:
                            grad_vfe_Wout = torch.mean(grad_vfe_Wout, dim=1)
                        self.Wout.weight += -alpha_lr * grad_vfe_Wout
                        tot_grad_vfe_Wout += -alpha_lr * grad_vfe_Wout


                    if self.use_input_bias:
                        grad_vfe_Wr_bias = self.Sigma_x * err_x_pre_sigma * (-1)
                        # If batch version is used, compute the mean gradient
                        if len(grad_vfe_Wr_bias.size()) == 3:
                            grad_vfe_Wr_bias = torch.mean(grad_vfe_Wr_bias, dim=0)
                        self.Wr.bias += -alpha_lr * (grad_vfe_Wr_bias).reshape(
                            (grad_vfe_Wr_bias).shape[1]
                        )
                        tot_grad_vde_b += -alpha_lr * (grad_vfe_Wr_bias).reshape(
                            (grad_vfe_Wr_bias).shape[1]
                        )

                    if self.use_output_bias:
                        grad_vfe_Wout_bias = self.Sigma_y * err_y_pre_sigma * (-1)
                        # If batch version is used, compute the mean gradient
                        if len(grad_vfe_Wout_bias.size()) == 3:
                            grad_vfe_Wout_bias = torch.mean(grad_vfe_Wout_bias, dim=0)
                        self.Wout.bias += -alpha_lr * (grad_vfe_Wout_bias).reshape(
                            (grad_vfe_Wout_bias).shape[1]
                        )
                        tot_grad_vde_bout += -alpha_lr * (grad_vfe_Wout_bias).reshape(
                            (grad_vfe_Wout_bias).shape[1]
                        )

                    if self.learn_Win and self.use_known_input:
                        expanded_error_vec = err_x_after_sigma.expand(
                            batch_size, seq_len, self.hidden_size
                        )
                        grad_vfe_Win = expanded_error_vec.mT @ (-u)
                        # If batch version is used, compute the mean gradient
                        if len(grad_vfe_Win.size()) == 3:
                            grad_vfe_Win = torch.mean(grad_vfe_Win, dim=0)
                        self.Win.weight += -alpha_lr * grad_vfe_Win
                        tot_grad_vfe_Win += -alpha_lr * grad_vfe_Win

                if self.learn_Wr and self.save_weight_history:
                    self.delta_Wr.append(tot_grad_vfe_Wr)
                    self.Wr_history.append(self.Wr.weight.data.clone())

                if self.learn_Wout and self.save_weight_history:
                    self.delta_Wout.append(tot_grad_vfe_Wout)
                    self.Wout_history.append(self.Wout.weight.data.clone())

                if self.use_input_bias and self.save_weight_history:
                    self.delta_b.append(tot_grad_vde_b)
                    self.b_history.append(self.Wr.bias.data.clone())

                if self.use_output_bias and self.save_weight_history:
                    self.delta_bout.append(tot_grad_vde_bout)
                    self.bout_history.append(self.Wout.bias.data.clone())

                if self.learn_Win and self.save_weight_history and self.use_known_input:
                    self.delta_Win.append(tot_grad_vfe_Win)
                    self.Win_history.append(self.Win.weight.data.clone())

            else:
                raise ValueError("Unknown mode: {}".format(mode))

        return self.x, self.Sigma_y, err_x_after_sigma, err_y




















