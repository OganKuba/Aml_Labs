from real_dataset import load_real_binary_dataset
from synthetic_dataset import generate_synthetic_binary_dataset


def main():
    # === Load Real Dataset ===
    print("Loading real-world dataset...")
    X_real, y_real = load_real_binary_dataset()
    print("Real dataset:")
    print(X_real.head())
    print(y_real.head())

    # === Generate Synthetic Dataset ===
    alpha = 0.0
    b = 1.0
    k = 5
    n = 1000
    random_state = 42

    print("\nGenerating synthetic dataset...")
    X_syn, y_syn = generate_synthetic_binary_dataset(alpha, b, k, n, random_state=random_state)
    print("Synthetic dataset:")
    print(X_syn.head())
    print(y_syn.head())


if __name__ == "__main__":
    main()
