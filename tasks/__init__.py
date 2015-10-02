from base import Classification

import mnist
import cluttered_mnist_video
import kth
import svhn
import goodfellow_svhn

def get_task(task_name, hyperparameters, **kwargs):
    klass = dict(mnist=mnist.Task,
                 cluttered_mnist_video=cluttered_mnist_video.Task,
                 kth=kth.Task,
                 svhn_digit=svhn.DigitTask,
                 svhn_number=goodfellow_svhn.NumberTask)[task_name]
    return klass(**hyperparameters)
