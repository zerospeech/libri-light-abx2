# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Callable, Union

import numpy as np
import torch

import libri_light_dtw as dtw
from .ABXIterators.abx_iterators_anycontext import *
from .ABXIterators.abx_iterators import *
from .models import Pooling


def get_distance_function_from_name(name_str: str):
    if name_str in ('euclidian', 'euclidean'):
        return get_euclidian_distance_batch
    if name_str == 'cosine':
        return get_cosine_distance_batch
    if name_str == 'kl':
        return get_kl_distance_batch
    if name_str == 'kl_symmetric':
        return get_kl_distance_symmetric_batch
    raise ValueError(f"Invalid distance mode")


def check_dtw_group_validity(a, b, x):
    assert(len(a.size()) == len(b.size()))
    assert(len(a.size()) == len(x.size()))
    assert(a.size(2) == x.size(2))
    assert(a.size(2) == b.size(2))

def get_kl_distance_batch(a1: torch.Tensor, a2: torch.Tensor, epsilon=1e-6):
    N1, S1, D = a1.size()  # Batch x Seq x Channel
    N2, S2, D = a2.size()  # Batch x Seq x Channel

    # (P * (P / Q).log()).sum()
    div = (a1.view(N1, 1, S1, 1, D) + epsilon) / (a2.view(1, N2, 1, S2, D) + epsilon)
    prod: torch.Tensor = (a1.view(N1, 1, S1, 1, D)) * div.log()

    return prod.sum(dim=4)

def get_kl_distance_symmetric_batch(a1: torch.Tensor, a2: torch.Tensor, epsilon=1e-6):
    N1, S1, D = a1.size()
    N2, S2, D = a2.size()

    div1 = (a1.view(N1, 1, S1, 1, D) + epsilon) / (a2.view(1, N2, 1, S2, D) + epsilon)
    div2 = (a2.view(1, N2, 1, S2, D) + epsilon) / (a1.view(N1, 1, S1, 1, D) + epsilon)

    prod1 = (a1.view(N1, 1, S1, 1, D)) * div1.log()
    prod2 = (a2.view(1, N2, 1, S2, D)) * div2.log()
    r: torch.Tensor = (0.5*prod1 + 0.5*prod2).sum(dim=4)
    return r

def get_cosine_distance_batch(a1: torch.Tensor, a2: torch.Tensor, epsilon=1e-6): # epsilon unused
    r""" a1 and a2 must be normalized"""
    N1, S1, D = a1.size()  # Batch x Seq x Channel
    N2, S2, D = a2.size()  # Batch x Seq x Channel

    prod = (a1.view(N1, 1, S1, 1, D)) * (a2.view(1, N2, 1, S2, D))
    # Sum accross the channel dimension
    prod = torch.clamp(prod.sum(dim=4), -1, 1).acos() / math.pi

    return prod


def get_euclidian_distance_batch(a1: torch.Tensor, a2: torch.Tensor, epsilon=1e-6): # epsilon unused
    N1, S1, D = a1.size()
    N2, S2, D = a2.size()
    diff = a1.view(N1, 1, S1, 1, D) - a2.view(1, N2, 1, S2, D)
    return torch.sqrt((diff**2).sum(dim=4))


def get_distance_group_dtw(a1: torch.Tensor, a2: torch.Tensor, size1: torch.Tensor, size2: torch.Tensor,
                           pooling: Pooling, ignore_diag=False, symmetric=False,
                           distance_function = get_cosine_distance_batch) -> torch.Tensor:

    N1, S1, D = a1.size()
    N2, S2, D = a2.size()
    if size1.size(0) != N1:
        print(a1.size(), size1.size())
        print(a2.size(), size2.size())
    assert(size1.size(0) == N1)
    assert(size2.size(0) == N2)
    distance_mat: np.ndarray = distance_function(a1, a2).detach().cpu().numpy()
    return dtw.dtw_batch(a1, a2, size1, size2,
                         distance_mat,
                         ignore_diag, symmetric)
    # TODO: We call dtw even when we have pooled and no realignment is needed. 
    # This should be harmless, but it is unnecessary. 
    # We should do something like (but add code to check that always the right 
    # dimensions are returned):
    # if pooling == Pooling.NONE:
    #    return dtw.dtw_batch(a1, a2, size1, size2,
    #                         distance_mat,
    #                         ignore_diag, symmetric)
    #Recast and reshape to what we would get from the dtw function if given pooled input
    #return torch.from_numpy(distance_mat.squeeze((1,2))) # Fix dims!

def get_theta_group_dtw(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
                        sa: torch.Tensor, sb: torch.Tensor, sx: torch.Tensor,
                        distance_function: Callable[..., torch.Tensor],
                        symmetric: bool, pooling: Pooling):

    check_dtw_group_validity(a, b, x)

    dxb = get_distance_group_dtw(
        x, b, sx, sb, pooling, distance_function=distance_function)
    dxa = get_distance_group_dtw(x, a, sx, sa, pooling,
                                 ignore_diag=symmetric, symmetric=symmetric,
                                 distance_function=distance_function)

    Nx, Na = dxa.size()
    Nx, Nb = dxb.size()

    if symmetric:
        n_pos = Na * (Na - 1)
        max_val = dxb.max().item()
        for i in range(Na):
            dxa[i, i] = max_val + 1
    else:
        n_pos = Na * Nx

    dxb = dxb.view(Nx, 1, Nb).expand(Nx, Na, Nb)
    dxa = dxa.view(Nx, Na, 1).expand(Nx, Na, Nb)

    sc = (dxa < dxb).sum() + 0.5 * (dxa == dxb).sum()
    sc /= (n_pos * Nb)

    return sc.item()


def loc_dtw(data,
            distance_function: Callable[..., torch.Tensor],
            symmetric: bool, pooling: Pooling):
    coords, group_a, group_b, group_x = data
    group_a_data, group_a_size = group_a
    group_b_data, group_b_size = group_b
    group_x_data, group_x_size = group_x
    theta = get_theta_group_dtw(group_a_data,
                                group_b_data,
                                group_x_data,
                                group_a_size,
                                group_b_size,
                                group_x_size,
                                distance_function,
                                symmetric,
                                pooling)

    return (coords, 1 - theta)

# TODO: maybe define a protocol for group_iterator (it can be 
# 4 different classes right now)
def get_abx_scores_dtw_on_group(group_iterator: Union[ABXWithinGroupIterator, ABXAcrossGroupIterator],
                                distance_function: Callable[..., torch.Tensor],
                                symmetric: bool, pooling: Pooling):

    data_list = []
    coords_list = []

    with torch.no_grad():
        # See the iterator's def __iter__(self) for details
        for _, group in enumerate(group_iterator):

            coords, abx = loc_dtw(group, distance_function, symmetric, pooling)
            data_list.append(abx)
            coords_list.append(coords)

    return torch.sparse.FloatTensor(torch.LongTensor(coords_list).t(),
                                    torch.FloatTensor(data_list),
                                    group_iterator.get_board_size())