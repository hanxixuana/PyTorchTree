#!/usr/bin/env python

"""
The functions as a whole give a solution to fitting a complete regression binary tree only using PyTorch tensors.
1. use init_tree to make two tensors to hold the tree split information and predictions at tree leaves
2. use build_tree to fill the two tensors from init_tree
3. use forward_tree to make predictions
"""

__author__ = "Xixuan Han"
__copyright__ = "Copyright (C) 2017 Xixuan Han"
__license__ = "MIT License"
__version__ = "1.0"
__print_split_info__ = False

import torch
import numpy as np


def search_optimal_split(data, target, pct_ignoring_for_two_side=0.6):
    """
    Search for the optimal split by minimizing the squared error.

    :param torch.FloatTensor data:          [n_row, n_col]
    :param torch.FloatTensor target:        [n_row, 1]
    :param float pct_ignoring_for_two_side: pct of rows at the beginning and end
                                            to be ignored for searching
    :return list:                           split summary

    Note:
    0.3 ** 2.0 * 24 = 2.16
    0.3 ** 3.0 * 75 = 2.025
    0.3 ** 4.0 * 250 = 2.025
    0.3 ** 5.0 * 850 = 2.065

    Example:

    import torch
    import numpy as np
    import build_tree as bt

    # gpu
    data = torch.cuda.FloatTensor(500, 90000).normal_()
    coefficients = torch.cuda.FloatTensor(90000, 1).normal_()
    target = data.mm(coefficients)
    %timeit bt.search_optimal_split(data, target)

    # cpu
    data = torch.FloatTensor(500, 90000).normal_()
    coefficients = torch.FloatTensor(90000, 1).normal_()
    target = data.mm(coefficients)
    %timeit bt.search_optimal_split(data, target)

    """
    n_row = data.size(0)
    n_col = data.size(1)
    _, sorted_row_idx = data.sort(0)

    # determine rows to be ignored

    left_start_row_idx = int(n_row * pct_ignoring_for_two_side / 2.0)

    if left_start_row_idx > 0:
        # prepare
        left_rows_to_be_ignored = target.expand(n_row, n_col).gather(
            0,
            sorted_row_idx[:left_start_row_idx]
        )
        rest_rows = target.expand(n_row, n_col).gather(
            0,
            sorted_row_idx[left_start_row_idx:]
        )
        # left
        left_cum_sum_per_col = left_rows_to_be_ignored.sum(0)
        left_cum_se_per_col = (
            left_rows_to_be_ignored.pow(2.0).sum(0)
            -
            left_rows_to_be_ignored.sum(0).pow(2.0).div(left_start_row_idx)
        )
        # right
        right_cum_sum_per_col = rest_rows.sum(0)
        right_cum_se_per_col = (
            rest_rows.pow(2.0).sum(0)
            -
            rest_rows.sum(0).pow(2.0).div(n_row - left_start_row_idx)
        )

    else:
        # left
        left_cum_sum_per_col = data.new(1, n_col).fill_(0.0)
        left_cum_se_per_col = data.new(1, n_col).fill_(0.0)
        # right
        grad_sum = target.sum()
        right_cum_sum_per_col = data.new(1, n_col).fill_(
            grad_sum
        )
        right_cum_se_per_col = data.new(1, n_col).fill_(
            target.sub(grad_sum / n_row).pow_(2.0).sum()
        )

    # for recording the optimal split

    smallest_loss = np.inf
    smallest_loss_feature_idx = np.random.randint(0, n_col)
    smallest_loss_feature_split_value = data[0, smallest_loss_feature_idx]
    smallest_loss_row_idx = 0

    left_pred = target[0, 0]
    right_pred = target[0, 0]

    flag = False

    for row_idx_from_left in xrange(left_start_row_idx, n_row - left_start_row_idx - 1):

        row_idx_from_right = n_row - row_idx_from_left - 1

        this_row = target.gather(0, sorted_row_idx[row_idx_from_left].unsqueeze(1)).t()

        squared_this_row = this_row.pow(2.0)

        if row_idx_from_left > 0:
            left_cum_se_per_col.add_(
                squared_this_row
                -
                (left_cum_sum_per_col + this_row).pow(2.0).div_(row_idx_from_left + 1)
                +
                left_cum_sum_per_col.pow(2.0).div_(row_idx_from_left)
            )
        if row_idx_from_right > 0:
            right_cum_se_per_col.sub_(
                squared_this_row
                -
                right_cum_sum_per_col.pow(2.0).div_(row_idx_from_right + 1)
                +
                (right_cum_sum_per_col - this_row).pow(2.0).div_(row_idx_from_right)
            )

        left_cum_sum_per_col.add_(
            this_row
        )
        right_cum_sum_per_col.sub_(
            this_row
        )

        cum_se_per_col = left_cum_se_per_col + right_cum_se_per_col
        smallest_cum_se_per_col, smallest_cum_se_feature_idx = cum_se_per_col.min(1)

        if smallest_loss > smallest_cum_se_per_col[0, 0]:

            flag = True
            smallest_loss = smallest_cum_se_per_col[0, 0]

            if n_row > 2:
                smallest_loss_feature_idx = smallest_cum_se_feature_idx[0, 0]
            else:
                smallest_loss_feature_idx = np.random.randint(0, n_col)

            smallest_loss_row_idx = row_idx_from_left

            left_pred = left_cum_sum_per_col[0, smallest_loss_feature_idx] / (row_idx_from_left + 1)
            right_pred = right_cum_sum_per_col[0, smallest_loss_feature_idx] / row_idx_from_right

    if flag:
        smallest_loss_feature_split_value = (
            (
                data[
                    sorted_row_idx[
                        smallest_loss_row_idx,
                        smallest_loss_feature_idx
                    ],
                    smallest_loss_feature_idx
                ]
                +
                data[
                    sorted_row_idx[
                        smallest_loss_row_idx + 1,
                        smallest_loss_feature_idx
                    ],
                    smallest_loss_feature_idx
                ]
            )
            /
            2.0
        )

    if __print_split_info__:
        print(
            {
                'n_sample': data.size(0),
                'n_feature': data.size(1),
                'smallest_loss_row_idx': smallest_loss_row_idx,
                'smallest_loss': smallest_loss,
            }
        )
        print(
            {
                'smallest_loss_feature_idx': smallest_loss_feature_idx,
                'smallest_loss_feature_split_value': smallest_loss_feature_split_value,
                'left_pred': left_pred,
                'right_pred': right_pred
            }
        )
        print('\n')

    return [
        smallest_loss_feature_idx,
        smallest_loss_feature_split_value,
        left_pred,
        right_pred,
        sorted_row_idx,
        smallest_loss_row_idx
    ]


def build_tree(tree_info, tree_pred, data, target, depth_idx, max_depth, node_idx):
    """
    Build a tree to tree_info and tree_pred.

    :param torch.FloatTensor tree_info:     [n ^ max_depth - 1, 5]
    :param torch.FloatTensor tree_pred:     [n ^ max_depth, ]
    :param torch.FloatTensor data:          [n_sample, n_feature]
    :param torch.FloatTensor target:        [n_sample, 1]
    :param int depth_idx:                   depth index
    :param int max_depth:                   max depth
    :param int node_idx:                    node index

    Example:

    import torch
    import numpy as np
    import build_tree as bt

    # gpu
    data = torch.cuda.FloatTensor(500, 90000).normal_()
    coefficients = torch.cuda.FloatTensor(90000, 1).normal_()
    target = data.mm(coefficients)

    max_depth = 4
    tree_info = torch.cuda.FloatTensor(2 ** max_depth - 1, 5)
    tree_pred = torch.cuda.FloatTensor(2 ** max_depth)

    %timeit bt.build_tree(tree_info, tree_pred, data, target, 0, max_depth, 0)

    # cpu
    torch.set_num_threads(12)
    data = torch.FloatTensor(500, 90000).normal_()
    coefficients = torch.FloatTensor(90000, 1).normal_()
    target = data.mm(coefficients)

    max_depth = 4
    tree_info = torch.FloatTensor(2 ** max_depth - 1, 5)
    tree_pred = torch.FloatTensor(2 ** max_depth)

    %timeit bt.build_tree(tree_info, tree_pred, data, target, 0, max_depth, 0)

    """

    if depth_idx < max_depth - 1:

        result = search_optimal_split(data, target)

        tree_info[node_idx, 0] = result[0]
        tree_info[node_idx, 1] = result[1]
        tree_info[node_idx, 2] = 0.0
        tree_info[node_idx, 3] = 2 * node_idx + 1
        tree_info[node_idx, 4] = 2 * node_idx + 2

        left_start_idx = result[5] + 1
        build_tree(
            tree_info,
            tree_pred,
            torch.index_select(
                data,
                0,
                result[4][:left_start_idx, result[0]]
            ),
            torch.index_select(
                target,
                0,
                result[4][:left_start_idx, result[0]]
            ),
            depth_idx + 1,
            max_depth,
            2 * node_idx + 1
        )

        if left_start_idx < result[4].size(0):
            right_start_idx = left_start_idx
        else:
            right_start_idx = left_start_idx - 1

        build_tree(
            tree_info,
            tree_pred,
            torch.index_select(
                data,
                0,
                result[4][right_start_idx:, result[0]]
            ),
            torch.index_select(
                target,
                0,
                result[4][right_start_idx:, result[0]]
            ),
            depth_idx + 1,
            max_depth,
            2 * node_idx + 2
        )

    else:

        result = search_optimal_split(data, target)

        row_idx_offset = 2 ** max_depth - 1

        tree_info[node_idx, 0] = result[0]
        tree_info[node_idx, 1] = result[1]
        tree_info[node_idx, 2] = 1.0
        tree_info[node_idx, 3] = 2 * node_idx + 1 - row_idx_offset
        tree_info[node_idx, 4] = 2 * node_idx + 2 - row_idx_offset

        tree_pred[2 * node_idx + 1 - row_idx_offset] = result[2]
        tree_pred[2 * node_idx + 2 - row_idx_offset] = result[3]


def forward_tree(x, tree_info, tree_pred):
    """
    Pass through the tree parallelly for samples.
    :param torch.FloatTensor x:         2D FloatTensor of [n_sample, n_feature]
    :param torch.FloatTensor tree_info: [n ^ max_depth - 1, 5]
    :param torch.FloatTensor tree_pred: [n ^ max_depth, ]
    :return torch.FloatTensor result:   2D FloatTensor of [n_sample, 1]
    """
    row_depth = x.new(x.size(0)).long()
    row_depth.fill_(0)

    max_depth = np.log2(tree_pred.size(0)).__long__()

    for depth_idx in xrange(max_depth):
        split_feature_idx = torch.gather(tree_info[:, 0], 0, row_depth).unsqueeze(1).long()

        split_candidate = torch.gather(x, 1, split_feature_idx)[:, 0]
        split_value = torch.gather(tree_info[:, 1], 0, row_depth)

        left_depth = torch.gather(tree_info[:, 3], 0, row_depth)
        right_depth = torch.gather(tree_info[:, 4], 0, row_depth)

        row_depth = (
            (split_candidate <= split_value).float() * left_depth
            +
            (split_candidate > split_value).float() * right_depth
        ).long()

    result = torch.gather(tree_pred, 0, row_depth).unsqueeze(1)

    return [
        result,
        row_depth
    ]


def init_tree(max_depth, device_idx=-1):
    """
    Initialize torch.FloatTensor tree_info of [n ^ max_depth - 1, 5]
    and torch.FloatTensor tree_pred of [n ^ max_depth, ].
    :param int max_depth:   max depth
    :param int device_idx:  device index
    :return list:           [tree_info, tree_pred]

    An Example:

    feature: x, y, z
    max_depth = 3

    tree_info:
        0               1           2       3           4
        split_feature   split_value leaf    left_node   right_node
    0   x               x_0         False   1           2

    1   y               y_1         False   3           4
    2   z               z_2         False   5           6

    3   x               x_3         True    0           1
    4   y               y_4         True    2           3
    5   x               x_5         True    4           5
    6   z               z_6         True    6           7

    tree_pred:
        0
    0   left_pred/right_pred
    1   a
    2   b
    3   c
    4   d
    5   e
    6   f
    7   g

    """
    if device_idx >= 0:
        tree_info = torch.cuda.FloatTensor(
            2 ** max_depth - 1,
            5,
            device=device_idx
        )
        tree_pred = torch.cuda.FloatTensor(
            2 ** max_depth,
            device=device_idx
        )
    else:
        tree_info = torch.FloatTensor(
            2 ** max_depth - 1,
            5
        )
        tree_pred = torch.FloatTensor(
            2 ** max_depth
        )
    return [
        tree_info,
        tree_pred
    ]


def test(seed=0, make_figure=True):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    max_depth = 4
    n_feature = 2
    n_sample = 500
    cuda_idx = 0

    data = torch.cuda.FloatTensor(n_sample, n_feature, device=cuda_idx).normal_()
    target = (
        (
            data - data.new(1, data.size(1)).normal_().div_(100.0).expand_as(data)
        ).pow(2.0).sum(1)
    )
    target.add_(
        target.new(target.size()).normal_() / 10.0
    )

    tree_info, tree_pred = init_tree(max_depth, cuda_idx)

    build_tree(tree_info, tree_pred, data, target, 0, max_depth, 0)

    output, _ = forward_tree(data, tree_info, tree_pred)

    mae = (target - output).abs().mean()

    print(['mae', mae])

    if make_figure:

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            data[:, 0].cpu().numpy(),
            data[:, 1].cpu().numpy(),
            target[:, 0].cpu().numpy(),
            alpha=0.3,
            s=5
        )

        ax.scatter(
            data[:, 0].cpu().numpy(),
            data[:, 1].cpu().numpy(),
            output[:, 0].cpu().numpy(),
            alpha=0.3,
            color='r',
            s=5
        )

        ax.set_xlabel('data[:, 0]')
        ax.set_ylabel('data[:, 1]')
        ax.set_zlabel('target')


if __name__ == '__main__':

    test()

    import matplotlib.pyplot as plt
    plt.show()
