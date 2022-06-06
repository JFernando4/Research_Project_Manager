import os

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_RUNNER = os.path.join(ROOT, "experiments", "experiment_runner.py")
MNIST_DATA_PATH = os.path.join(ROOT, "problems", "supervised_learning", "mnist")
CIFAR_DATA_PATH = os.path.join(ROOT, "problems", "supervised_learning", "cifar")
