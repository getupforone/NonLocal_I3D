#!/usr/bin/env python3
import torch

def topks_correct(preds, labels, ks):
    assert preds.size(0) == labels.size(
        0
    ),"Batch dim of predictions and labels must match"

    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )

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