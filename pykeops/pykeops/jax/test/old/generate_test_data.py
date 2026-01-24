import numpy as np


def generate_data():
    N, M, D, B = 100, 80, 3, 4
    np.random.seed(42)  # Shared seed

    data = {
        "x_2d": np.random.randn(N, D).astype('float32'),
        "y_2d": np.random.randn(M, D).astype('float32'),
        "x_3d": np.random.randn(B, N, D).astype('float32'),
        "y_3d": np.random.randn(B, M, D).astype('float32')
    }
    np.savez("test_data.npz", **data)
    print("✅ Reference data saved to test_data.npz")


if __name__ == "__main__":
    generate_data()