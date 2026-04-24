# The-Self-Pruning-Neural-Network

You will build a standard feed-forward neural network for an image classification task
(e.g., on CIFAR-10). However, you must augment this network with a mechanism that
allows it to learn which of its own weights are unnecessary.
The core idea is to associate each weight in the network with a learnable "gate"
parameter. This gate, a single scalar value between 0 and 1, will multiply the weight's
output. If a gate's value becomes 0, it effectively "prunes" or removes the corresponding
weight from the network.
Your goal is to formulate a loss function and training procedure that encourages most of
these gates to become exactly zero, leaving only a "sparse" network of the most
important connections.
