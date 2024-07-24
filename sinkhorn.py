import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Inputs:
        - lam : strength of the entropic regularization
        - max_iter (int): maximum number of Sinkhorn iterations
        - reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Middle:
        - C : cost matrix (shape = n x m)
        - r : vector of marginals (shape = n x 1)
        - c : vector of marginals (shape = m x 1)
    Output:
        - P : optimal transport matrix (n x m)
        - cost : Sinkhorn distance
    """

    def __init__(self, lamda, max_iter=1000, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.lamda = lamda
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y, margin_mu, margin_nu, stopThr=1e-7, device=None):
        # The Sinkhorn algorithm takes as input four variables:
        # margin_mu and margin_nu are measures
        x_points = x.size(0)
        y_points = y.size(0)

        C = self.cost_matrix(x, y)  # Wasserstein cost function, (n,m)
        K = torch.exp(- C * self.lamda).to(device)
        a = margin_mu.sum().repeat(margin_mu.size(0), 1).to(device)
        a /= a.sum()

        displacement_square_norm = stopThr + 1.
        iter_count = 0
        T_sum = torch.zeros((x_points, y_points)).to(device)
        while displacement_square_norm > stopThr and iter_count < self.max_iter:
            b = margin_nu / torch.matmul(torch.t(K), a)
            a = margin_mu / torch.matmul(K, b).float()
            P = torch.matmul(torch.matmul(torch.diag(a[:, 0]), K), torch.diag(b[:, 0]))
            displacement_square_norm = torch.sum(torch.square(T_sum - P))
            T_sum = P
            iter_count += 1

        cost = torch.sum(torch.mul(P, C))
        return cost, P, C

    @staticmethod
    def cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C