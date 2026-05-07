import matplotlib.pyplot as plt
from mnist_ann import train_and_evaluate as ann_model
from mnist_cnn import train_and_evaluate as cnn_model
ann_acc , ann_losses  = ann_model()
cnn_acc , cnn_losses = cnn_model()
models = ['ANN', 'CNN']
accuracies = [ann_acc, cnn_acc]
plt.bar(models, accuracies)
plt.ylabel("Accuracy (%)")
plt.title("ANN vs CNN on MNIST")
plt.savefig("comparison.png")
plt.show()