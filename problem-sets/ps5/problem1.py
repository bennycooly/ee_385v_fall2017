
from mlp import MLP

def main():
    mlp = MLP("mnistabridged.mat", 25)
    mlp.train()


if __name__ == "__main__":
    main()
