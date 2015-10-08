from base import Classification

import mnist
import cmv
import kth
import svhn
import goodfellow_svhn

def get_task(task_name, hyperparameters, **kwargs):
    klass = dict(mnist=mnist.Task,
                 cmv=cmv.Task,
                 kth=kth.Task,
                 svhn_digit=svhn.DigitTask,
                 svhn_number=goodfellow_svhn.NumberTask)[task_name]
    return klass(**hyperparameters)
