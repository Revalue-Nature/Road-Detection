import numpy as np
import tensorflow as tf
from abc import ABCMeta,abstractmethod

from utils import eval_methods as EM
from utils import loss_tf as LF
from utils import util as U
from utils.process_methods import one_hot

class Model(metaclass=ABCMeta):
    
    def __init__(self, net):
        self.net = net

    @abstractmethod
    def get_grads(self, data_dict):
        """
        """

    @abstractmethod
    def eval(self, data_dict, **kwargs):
        """
        """

    @abstractmethod
    def predict(self, data_dict):
        """
        """



class SimpleTFModel(Model):
    def __init__(self, net, x_suffix, y_suffix=None, m_suffix=None, dropout=0, loss_function={'cross-entropy': 0.5, 'dice':0.5}):
        super().__init__(net)
        self._x_suffix = x_suffix
        self._y_suffix = y_suffix
        self._m_suffix = m_suffix
        self._loss_function = loss_function

        self._loss_dict = {'cross-entropy': LF.cross_entropy,
                           'dice': LF.dice_coefficient}

        self.dropout = dropout

    def get_grads(self, data_dict):
        xs = data_dict[self._x_suffix]

        with tf.GradientTape() as tape:
            logits = self.net(xs, self.dropout, True)
            loss = self._get_loss(logits, data_dict)                       
        grads = tape.gradient(loss, self.net.trainable_variables)
        return grads

    def eval(self, data_dict, **kwargs):
        xs = data_dict[self._x_suffix]
        ys = data_dict[self._y_suffix]
        # ms = data_dict[self._m_suffix]
        logits = self.net(xs, 0, False)
        prob = tf.nn.softmax(logits, -1)
        pred = one_hot(np.argmax(prob, -1), list(range(ys.shape[-1])))
        # pred = np.argmax(prob, -1)
        ys = one_hot(ys, list(range(ys.shape[-1])))
        loss = self._get_loss(logits, data_dict)
        # loss = tf.reduce_mean(loss, range(1, loss.ndim))
        if loss.ndim > 1:
            loss = tf.reduce_mean(loss, range(1, loss.ndim))
        else:
            loss = tf.reduce_mean(loss)

        loss = [loss] if loss.ndim == 0 else loss


        pred = np.resize(pred, (3, 128, 128, 2) )
        ys = np.resize(ys, (3, 128, 128, 2))
        # pred =
        ys = np.transpose(ys, (1, 2, 0, 3))
        pred = np.transpose(pred, (1, 2, 0, 3))


        pred = np.expand_dims(pred, 0)
        ys = np.expand_dims(ys, 0)
        # pred = np.reshape (pred, [pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]])
        # ys = np.reshape (ys, [ys.shape[0], ys.shape[1], ys.shape[2], ys.shape[3]])


        acc = EM.accuracy(pred, ys)
        dice = EM.dice_coefficient(pred, ys)
        iou = EM.iou(pred, ys)
        # auc = EM.auc(pred, ys)

        precision = EM.precision(pred, ys)
        recall = EM.recall(pred, ys)
        sensitivity = EM.sensitivity(pred, ys)
        specificity = EM.specificity(pred, ys)

        
        eval_results = {'loss': loss,
                    'acc': acc,
                    'dice': dice,
                    'iou': iou,
                    # 'auc': auc,
                    'precision': precision,
                    'recall': recall,
                    'sensitivity': sensitivity,
                    'specificity': specificity}

        need_imgs = kwargs.get('need_imgs', False)
        if need_imgs:
            imgs = self._get_imgs_eval(xs, ys, prob)
            eval_results.update({'imgs': imgs})

        need_logits = kwargs.get('need_logits', False)
        if need_logits:
            eval_results.update({'logits': logits})

        need_preds = kwargs.get('need_preds', False)
        if need_preds:
            eval_results.update({'pred': pred})

        return eval_results

    def predict(self, data_dict):
        logits = self.net(data_dict[self._x_suffix], 0, False)
        prob = tf.nn.softmax(logits, -1)
        return prob

    def _get_loss(self, logits, data_dict):
        loss_data_dict = {'orgs': data_dict[self._x_suffix],
                          'labels': data_dict[self._y_suffix],
                          'masks': None,
                          'logits': logits}
        total_loss = 0
        assert self._loss_function is not None and type(self._loss_function) is dict, 'loss_function should be a dict ({name: weight})'
        for lf in self._loss_function:
            weight = self._loss_function[lf]
            loss_function = self._loss_dict[lf]
            sub_loss_map = loss_function(loss_data_dict)
            # sub_loss = tf.reduce_mean(sub_loss_map, range(1, sub_loss_map.ndim))
            if sub_loss_map.ndim > 1:
                sub_loss = tf.reduce_mean(sub_loss_map, range(1, sub_loss_map.ndim))
            else:
                sub_loss = tf.reduce_mean(sub_loss_map)
            total_loss += sub_loss * weight

        return total_loss

    def _get_imgs_eval(self, xs, ys, prob):
        img_dict = {}
        n_class = ys.shape[-1]
        for i in range(n_class):
            img = U.combine_2d_imgs_from_tensor([xs[..., 0], ys[..., i], prob[..., i]])
            img_dict.update({'class %d'%i: img})

        argmax_ys = np.argmax(ys, -1)
        argmax_prob = np.argmax(prob, -1)
        img = U.combine_2d_imgs_from_tensor([xs[..., 0], argmax_ys, argmax_prob])
        img_dict.update({'argmax': img})

        return img_dict

