import cupy as cp
import time


def stress_test_gpu(size=8192, iterations=100):
    """
    Perform a stress test on the GPU using large matrix multiplications.

    Parameters:
    size (int): Size of the square matrices.
    iterations (int): Number of multiplications to perform.
    """
    print(
        f"Starting GPU stress test with {size}x{size} matrices for {iterations} iterations..."
    )

    # Allocate large matrices on GPU
    A = cp.random.randn(size, size, dtype=cp.float32)
    B = cp.random.randn(size, size, dtype=cp.float32)

    start_time = time.time()

    for i in range(iterations):
        C = A @ B  # Matrix multiplication
        cp.cuda.Device(0).synchronize()  # Ensure computation is complete

        if i % (iterations // 10) == 0:
            print(f"Iteration {i}/{iterations} completed")

    end_time = time.time()

    print(f"GPU stress test completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    stress_test_gpu()
