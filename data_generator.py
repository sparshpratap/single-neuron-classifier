import numpy as np

def generate_gaussian_data(
    modes_per_class=1,
    samples_per_mode=50,
    mean_range=(-1, 1),
    std_range=(0.1, 0.3)
):
    """
    Generates 2D Gaussian data for two classes (0 and 1).
    """

    X = []
    y = []

    for class_label in [0, 1]:
        for _ in range(modes_per_class):
            mean = np.random.uniform(mean_range[0], mean_range[1], size=2)
            std = np.random.uniform(std_range[0], std_range[1])

            samples = np.random.normal(loc=mean, scale=std,
                                        size=(samples_per_mode, 2))

            X.append(samples)
            y.append(np.full(samples_per_mode, class_label))

    X = np.vstack(X)
    y = np.concatenate(y)

    return X, y
