import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def CCA(view1, view2, num_shared_dim, r1=0, r2=0):
    V1 = tf.cast(view1, dtype=tf.float32)
    V2 = tf.cast(view2, dtype=tf.float32)

    assert V1.shape[0] == V2.shape[0]
    M = tf.cast(tf.shape(V1)[0], dtype=tf.float32)
    ddim_1 = tf.constant(V1.shape[1], dtype=tf.int16)
    ddim_2 = tf.constant(V2.shape[1], dtype=tf.int16)

    # check mean and variance
    mean_V1 = tf.reduce_mean(V1, 0)
    mean_V2 = tf.reduce_mean(V2, 0)

    V1_bar = tf.subtract(V1, tf.tile(tf.convert_to_tensor(mean_V1)[None], [M, 1]))
    V2_bar = tf.subtract(V2, tf.tile(tf.convert_to_tensor(mean_V2)[None], [M, 1]))

    Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
    Sigma11 = tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) + r1 * tf.eye(ddim_1)
    Sigma22 = tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) + r2 * tf.eye(ddim_2)

    Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
    Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
    Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

    C = tf.linalg.matmul(tf.linalg.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
    D, U, V = tf.linalg.svd(C, full_matrices=False)

    A = tf.matmul(tf.transpose(U)[:num_shared_dim], Sigma11_root_inv)
    B = tf.matmul(tf.transpose(V)[:num_shared_dim], Sigma22_root_inv)

    epsilon = tf.matmul(A, tf.transpose(V1_bar))
    omega = tf.matmul(B, tf.transpose(V2_bar))

    return A, B, epsilon, omega, D[:num_shared_dim], mean_V1, mean_V2


def PCC_Matrix(view1, view2, observations):
    assert tf.shape(view1)[1] == observations
    assert tf.shape(view1)[1] == tf.shape(view2)[1]
    calc_cov = tfp.stats.correlation(view1, view2, sample_axis=1, event_axis=0)

    return tf.math.abs(calc_cov), tf.shape(view1)[0], tf.shape(view2)[0]


def MAXVARCCA_single_iteration(view1, view2, view3, regularization=0):
    V1 = tf.cast(view1, dtype=tf.float32)
    V2 = tf.cast(view2, dtype=tf.float32)
    V3 = tf.cast(view3, dtype=tf.float32)

    assert V1.shape[0] == V2.shape[0] and V2.shape[0] == V3.shape[0]
    M = tf.constant(V1.shape[0], dtype=tf.float32)

    ddim_1 = tf.constant(V1.shape[1], dtype=tf.int16).numpy()
    ddim_2 = tf.constant(V2.shape[1], dtype=tf.int16).numpy()
    ddim_3 = tf.constant(V3.shape[1], dtype=tf.int16).numpy()

    # check mean and variance
    mean_V1 = tf.reduce_mean(V1, 0)
    mean_V2 = tf.reduce_mean(V2, 0)
    mean_V3 = tf.reduce_mean(V3, 0)

    V1_bar = tf.subtract(V1, tf.tile(tf.convert_to_tensor(mean_V1)[None], [M, 1]))
    V2_bar = tf.subtract(V2, tf.tile(tf.convert_to_tensor(mean_V2)[None], [M, 1]))
    V3_bar = tf.subtract(V3, tf.tile(tf.convert_to_tensor(mean_V3)[None], [M, 1]))

    R11 = tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) + regularization * tf.eye(ddim_1)
    R22 = tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) + regularization * tf.eye(ddim_2)
    R33 = tf.linalg.matmul(tf.transpose(V3_bar), V3_bar) / (M - 1) + regularization * tf.eye(ddim_3)

    RD = tf.linalg.LinearOperatorBlockDiag(
        [
            tf.linalg.LinearOperatorFullMatrix(R11),
            tf.linalg.LinearOperatorFullMatrix(R22),
            tf.linalg.LinearOperatorFullMatrix(R33),
        ]
    ).to_dense() + regularization * tf.eye(ddim_1 + ddim_2 + ddim_3)
    RD_root_inv = tf.linalg.sqrtm(tf.linalg.inv(RD))

    V = tf.concat([V1_bar, V2_bar, V3_bar], axis=1)
    R = tf.linalg.matmul(tf.transpose(V), V) / (M - 1)

    C = tf.linalg.matmul(tf.linalg.matmul(RD_root_inv, R), RD_root_inv)
    eigen_values, eigen_vectors = tf.linalg.eig(C)
    w_tilde_1 = tf.cast(eigen_vectors[:, -1][:, None], tf.float32)

    min_dim = tf.math.reduce_min([ddim_1, ddim_2, ddim_3]).numpy()
    ccor = tf.cast(eigen_values[-min_dim:][::-1], tf.float32) / 3

    W1 = tf.linalg.matmul(tf.transpose(w_tilde_1[0:ddim_1]), tf.linalg.inv(R11))
    W2 = tf.linalg.matmul(tf.transpose(w_tilde_1[ddim_1 : ddim_1 + ddim_2]), tf.linalg.inv(R22))
    W3 = tf.linalg.matmul(tf.transpose(w_tilde_1[ddim_1 + ddim_2 : ddim_1 + ddim_2 + ddim_3]), tf.linalg.inv(R33))

    epsilon = tf.linalg.matmul(W1, tf.transpose(V1_bar))
    omega = tf.linalg.matmul(W2, tf.transpose(V2_bar))
    theta = tf.linalg.matmul(W3, tf.transpose(V3_bar))

    # Normalize them by norm
    epsilon = epsilon / tf.linalg.norm(epsilon)
    omega = omega / tf.linalg.norm(omega)
    theta = theta / tf.linalg.norm(theta)

    return W1, W2, W3, epsilon, omega, theta, ccor, mean_V1, mean_V2, mean_V3


def MAXVARCCA(view1, view2, view3, regularization=0):
    # Transpose input data
    view1 = tf.transpose(view1)
    view2 = tf.transpose(view2)
    view3 = tf.transpose(view3)

    epsilon_list = []
    omega_list = []
    theta_list = []

    z_1 = []
    z_2 = []
    z_3 = []
    for i in range(view1.shape[0]):
        if len(z_1) == 0:
            # First time, no previous eigenvectors to consider
            tmp_view1 = tf.identity(view1)
            tmp_view2 = tf.identity(view2)
            tmp_view3 = tf.identity(view3)

        else:
            # Deflate input data
            view_1_matrix = tf.concat(z_1, axis=1)
            view_1_matrix_sq = tf.linalg.matmul(view_1_matrix, tf.transpose(view_1_matrix))
            tmp_view1 = tf.linalg.matmul(tf.identity(view1), tf.eye(view1.shape[1]) - view_1_matrix_sq)

            singular_values, left_sing_vec, right_sing_vec = tf.linalg.svd(tmp_view1)
            tmp_view1 = tf.linalg.matmul(tf.transpose(left_sing_vec[:, :-i]), tmp_view1)

            view_2_matrix = tf.concat(z_2, axis=1)
            view_2_matrix_sq = tf.linalg.matmul(view_2_matrix, tf.transpose(view_2_matrix))
            tmp_view2 = tf.linalg.matmul(tf.identity(view2), tf.eye(view2.shape[1]) - view_2_matrix_sq)

            singular_values, left_sing_vec, right_sing_vec = tf.linalg.svd(tmp_view2)
            tmp_view2 = tf.linalg.matmul(tf.transpose(left_sing_vec[:, :-i]), tmp_view2)

            view_3_matrix = tf.concat(z_3, axis=1)
            view_3_matrix_sq = tf.linalg.matmul(view_3_matrix, tf.transpose(view_3_matrix))
            tmp_view3 = tf.linalg.matmul(tf.identity(view3), tf.eye(view3.shape[1]) - view_3_matrix_sq)

            singular_values, left_sing_vec, right_sing_vec = tf.linalg.svd(tmp_view3)
            tmp_view3 = tf.linalg.matmul(tf.transpose(left_sing_vec[:, :-i]), tmp_view3)

        W1, W2, W3, epsilon, omega, theta, ccor, mean_V1, mean_V2, mean_V3 = MAXVARCCA_single_iteration(
            tf.transpose(tmp_view1), tf.transpose(tmp_view2), tf.transpose(tmp_view3), regularization
        )

        epsilon_list.append(epsilon)
        omega_list.append(omega)
        theta_list.append(theta)

        z_1.append(tf.transpose(epsilon) / tf.linalg.norm(epsilon))
        z_2.append(tf.transpose(omega) / tf.linalg.norm(omega))
        z_3.append(tf.transpose(theta) / tf.linalg.norm(theta))

    epsilon_total = tf.concat(epsilon_list, axis=0)
    omega_total = tf.concat(omega_list, axis=0)
    theta_total = tf.concat(theta_list, axis=0)

    corr_pair_1 = tf.abs(tf.linalg.diag_part(tf.matmul(epsilon_total, tf.transpose(omega_total))))
    corr_pair_2 = tf.abs(tf.linalg.diag_part(tf.matmul(omega_total, tf.transpose(theta_total))))
    corr_pair_3 = tf.abs(tf.linalg.diag_part(tf.matmul(theta_total, tf.transpose(epsilon_total))))

    all_corr_values = tf.concat([corr_pair_1, corr_pair_2, corr_pair_3], axis=0)

    return epsilon_total, omega_total, theta_total, all_corr_values
