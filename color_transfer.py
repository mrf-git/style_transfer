"""
Defines methods for transferring color.
"""

import numpy as np


def compute_mean_and_cov(arr):
    flattened = arr.reshape((-1, 3))
    mean = np.mean(flattened, axis=0)
    cov = np.zeros((3, 3), dtype="float32")
    for i in range(flattened.shape[0]):
        v = np.expand_dims(flattened[i] - mean, axis=1)
        cov += np.dot(v, v.T)
    cov /= flattened.shape[0]

    return mean, cov


def compute_cov_sqrt(cov):
    u, s, vh = np.linalg.svd(cov)
    s_root = np.sqrt(s)
    return np.dot(np.dot(u, np.diag(s_root)), vh)


def compute_color_transform(content_arr, style_arr, add_noise=False):
    if add_noise:
        content_arr = np.clip(content_arr + np.random.rand(*content_arr.shape).astype("float32"), 0, 255)
        mean_content, cov_content = compute_mean_and_cov(content_arr)
        style_arr = np.clip(style_arr + np.random.rand(*style_arr.shape).astype("float32"), 0, 255)
        mean_style, cov_style = compute_mean_and_cov(style_arr)
    else:
        mean_content, cov_content = compute_mean_and_cov(content_arr)
        mean_style, cov_style = compute_mean_and_cov(style_arr)

    cov_content_sqrt = compute_cov_sqrt(cov_content)
    cov_style_sqrt = compute_cov_sqrt(cov_style)

    A = np.dot(cov_content_sqrt, np.linalg.inv(cov_style_sqrt))
    b = mean_content - np.dot(A, mean_style)

    return A, b


def transfer_color_histogram_matching(source_arr, target_arr):
    try:
        A, b = compute_color_transform(source_arr, target_arr)
    except np.linalg.LinAlgError:
        A, b = compute_color_transform(source_arr, target_arr, True)

    out_arr = (np.dot(target_arr.reshape(-1, 3), A.T) + b).reshape(target_arr.shape)

    return np.clip(out_arr, 0, 255)


def transfer_color_luminance(source_arr, target_arr):
    M = np.array([[0.299, 0.587, 0.114],
                  [0.595716, -0.274453, -0.321263],
                  [0.211456, -0.522591, 0.311135]], dtype="float32")
    M_inv = np.array([[1, 0.9563, 0.621],
                      [1, -0.2721, -0.674],
                      [1, -1.107, 1.7046]], dtype="float32")

    yiq_source_arr = (np.dot(source_arr.reshape(-1, 3), M.T)).reshape(source_arr.shape)
    yiq_target_arr = (np.dot(target_arr.reshape(-1, 3), M.T)).reshape(target_arr.shape)

    yiq_target_arr[:, :, 1:] = yiq_source_arr[:, :, 1:]
    out_arr = (np.dot(yiq_target_arr.reshape(-1, 3), M_inv.T)).reshape(yiq_target_arr.shape)

    return np.clip(out_arr, 0, 255)
