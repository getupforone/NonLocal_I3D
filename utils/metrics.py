#!/usr/bin/env python3
import torch

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1,5] corresponds 
            to top-1 and top-5.
    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions
    """
    assert preds.size(0) == labels.size(
        0
    ),"Batch dim of predictions and labels must match"
    #print("preds shape  = {}".format(preds.shape))
    #print("preds shape  = {}".format(preds.shape))
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    #print("top_max_k_inds shape  = {}".format(top_max_k_inds.shape))

    top_max_k_inds = top_max_k_inds.t()
    rep_max_k_labels = labels.view(1,-1).expand_as(top_max_k_inds)
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    return topks_correct

def topk_errors(preds, labels, ks):
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]

def topk_accuracies(preds, labels, ks):
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0))* 100.0 for x in num_topks_correct]