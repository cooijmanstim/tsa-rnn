from base import Classification

import mnist
import old_cmv
import cmv
import kth
import ucf101
import svhn
import goodfellow_svhn

def get_task(task_name, hyperparameters, **kwargs):
    klass = dict(mnist=mnist.Task,
                 old_cmv=old_cmv.Task,
                 cmv=cmv.Task,
                 kth=kth.Task,
                 ucf101=ucf101.Task,
                 svhn_digit=svhn.DigitTask,
                 svhn_number=goodfellow_svhn.NumberTask)[task_name]
    return klass(**hyperparameters)
