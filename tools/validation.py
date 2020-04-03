import time

import numpy as np
import torch
from torch.nn import functional as F

from torchreid import metrics
from torchreid import utils
from torchreid.utils import avgmeter


def _parse_data_for_eval(data):
    return data[0], data[1], data[2]


def evaluate(model,
             query_loader=None,
             gallery_loader=None,
             dist_metric='euclidean',
             use_metric_cuhk03=False,
             ranks=[1, 5, 10, 20],
             rerank=False):
    batch_time = avgmeter.AverageMeter()

    def _feature_extraction(data_loader):
        f_, pids_, camids_ = [], [], []
        for batch_idx, data in enumerate(data_loader):
            imgs, pids, camids = _parse_data_for_eval(data)
            end = time.time()
            _, features = model.predict_on_batch(imgs)
            batch_time.update(time.time() - end)
            f_.extend(features.numpy())
            pids_.extend(pids.numpy())
            camids_.extend(camids.numpy())
        f_ = np.stack(f_)
        pids_ = np.stack(pids_)
        camids_ = np.stack(camids_)
        return f_, pids_, camids_

    print('Extracting features from query set ...')
    qf, q_pids, q_camids = _feature_extraction(query_loader)
    print('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))

    print('Extracting features from gallery set ...')
    gf, g_pids, g_camids = _feature_extraction(gallery_loader)
    print('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))

    print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

    # if normalize_feature:
    #     print('Normalzing features with L2 norm ...')
    #     qf = F.normalize(qf, p=2, dim=1)
    #     gf = F.normalize(gf, p=2, dim=1)

    print(f'Computing distance matrix with metric={dist_metric} ...')
    distmat = compute_distance_matrix(qf, gf, dist_metric)

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = compute_distance_matrix(qf, qf, dist_metric)
        distmat_gg = compute_distance_matrix(gf, gf, dist_metric)
        distmat = utils.re_ranking(distmat, distmat_qq, distmat_gg)

    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        use_metric_cuhk03=use_metric_cuhk03
    )

    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

    # if visrank:
    #     visualize_ranked_results(
    #         distmat,
    #         self.datamanager.
    #             return_query_and_gallery_by_name(dataset_name),
    #         self.datamanager.data_type,
    #         width=self.datamanager.width,
    #         height=self.datamanager.height,
    #         save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
    #         topk=visrank_topk
    #     )

    return cmc[0]


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (numpy.ndarray): 2-D feature matrix.
        input2 (numpy.ndarray): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        numpy.ndarray: distance matrix.

    Examples::
       >>> from torchreid import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2)
       >>> distmat.size() # (10, 100)
    """
    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (numpy.ndarray): 2-D feature matrix.
        input2 (numpy.ndarray): 2-D feature matrix.

    Returns:
        numpy.ndarray: distance matrix.
    """
    m, n = input1.shape[0], input2.shape[0]
    part1 = np.tile(np.power(input1, 2).sum(axis=1, keepdims=True), [1, n])
    part2 = np.tile(np.power(input2, 2).sum(axis=1, keepdims=True), [1, m]).transpose()
    distmat = part1 + part2
    result = 1 * distmat + -2 * (input1 @ input2.transpose())
    # distmat.addmm_(1, -2, input1, input2.t())
    return result


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (numpy.ndarray): 2-D feature matrix.
        input2 (numpy.ndarray): 2-D feature matrix.

    Returns:
        numpy.ndarray: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat
