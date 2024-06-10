import numpy as np
import matplotlib.pyplot as plt


def gaussian(u):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-u ** 2 / 2)


def kde(x_samples, h, K=gaussian):
    # データ数
    n = len(x_samples)

    X = np.linspace(min(x_samples) - 2, max(x_samples) + 2, 100)

    estimated = np.zeros(len(X))
    for x_sample in x_samples:
        func_output = []
        for x in X:
            k = K((x - x_sample) / h)
            func_output.append(k)
        estimated += np.array(func_output)

    estimated /= n * h
    return X, estimated


def plot_estimeted(X, estimated, x_samples):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(X, estimated)
    ax.plot(x_samples, [0] * len(x_samples), 'ro')
    fig.savefig('kde.png')


if __name__ == '__main__':
    # kernel density estimation

    # サンプル生成
    x_samples = [0.1, 2.3, 3.0, 4.7, 6.0, 6.2, 6.6, 7.1, 9.5]
    print(x_samples)

    # KDEのバンド幅を設定
    h = 0.5

    X, estimated = kde(x_samples, h)
    plot_estimeted(X, estimated, x_samples)
