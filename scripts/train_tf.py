import argparse
import glob
import re
import os
import sys

import numpy as np
import tensorflow as tf

from torchreid.models import osnet_tf
from torchreid.utils import utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--model-dir', default='train')
    parser.add_argument('--output', default='saved_model')
    parser.add_argument('--data-dir')
    parser.add_argument('--dataset-name')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)

    return parser.parse_args()


class Market1501Dataset:
    def __init__(self, data_dir, mode='train', batch_size=16):
        self.batch_size = batch_size

        relabel = mode == 'train'
        data_dir = data_dir.rstrip('/')
        img_paths = glob.glob(data_dir + '/bounding_box_train/*.jpg')
        test_img_paths = glob.glob(data_dir + '/bounding_box_test/*.jpg')
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        data = []

        paths = []
        pids = []
        test_paths = []
        test_pids = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid))
            paths.append(img_path)
            pids.append(pid)

        for img_path in test_img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            test_paths.append(img_path)
            test_pids.append(pid)

        self.paths = paths
        self.pids = pids
        self.test_paths = test_paths
        self.test_pids = test_pids

    def get_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.paths, self.pids))
        return dataset.map(preprocess).batch(self.batch_size).shuffle(self.batch_size * 2)

    def get_test_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.test_paths, self.test_pids))
        return dataset.map(preprocess).batch(self.batch_size)

    def num_classes(self):
        return len(set(self.pids))


mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])


def preprocess(image_path, label):
    raw_content = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw_content, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image to the desired size.
    image = tf.image.resize(image, [256, 128])
    image = image / 255.0
    image = (image - mean) / std
    # image = tf.image.resize(image, (256, 128))
    return image, label


class CrossEntropyLoss(tf.keras.losses.Loss):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        \begin{equation}
        (1 - \epsilon) \times y + \frac{\epsilon}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\epsilon` is a weight. When
    :math:`\epsilon = 0`, the loss function reduces to the normal cross entropy.

    Args:
        num_classes (int): number of classes.
        epsilon (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, num_classes, batch_size, epsilon=0.1, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.logsoftmax = tf.nn.log_softmax
        self.batch_size = batch_size

    def call(self, y_true, y_pred):
        """
        Args:
            y_pred (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            y_true (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(y_pred, axis=1)
        # targets = utils.scatter_numpy(zeros.numpy(), 1, tf.expand_dims(y_true, 1).numpy(), 1)
        expanded = tf.expand_dims(tf.squeeze(y_true), 1)
        expanded = tf.cast(expanded, tf.int32)
        # tf.scatter_nd([[0,0],[1,1],[2,2],[3,3],[4,4]], np.ones([5]), [5, 10])
        targets = tf.scatter_nd(
            tf.concat((tf.expand_dims(tf.range(self.batch_size), 1), expanded), axis=1),
            tf.ones([self.batch_size]),
            [self.batch_size, self.num_classes]
        )
        # if self.use_gpu:
        #     targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return tf.reduce_sum(tf.reduce_mean(-targets * log_probs, axis=0))


class Scheduler:
    def __init__(self, initial_learning_rate=0.05, epochs=10):
        self.epochs = epochs
        self.learning_rate = initial_learning_rate

    def schedule(self, epoch):
        if epoch <= 0.3 * self.epochs:
            return self.learning_rate
        elif epoch <= 0.5 * self.epochs:
            return self.learning_rate / 5
        elif epoch <= 0.75 * self.epochs:
            return self.learning_rate / 25
        else:
            return self.learning_rate / 250


def main():
    args = parse_args()
    dataset = Market1501Dataset(args.data_dir, args.mode, args.batch_size)
    model = osnet_tf.osnet_x0_25(num_classes=dataset.num_classes())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(amsgrad=True),
        loss=[CrossEntropyLoss(num_classes=dataset.num_classes(), batch_size=args.batch_size), None],
        # loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), None],
        metrics=['accuracy']
    )
    mode = args.mode

    if mode == 'train':
        scheduler = Scheduler(initial_learning_rate=args.lr, epochs=args.epochs)
        model.fit(
            x=dataset.get_input_fn(),
            # validation_data=dataset.get_test_input_fn(),
            # batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1 if sys.stdout.isatty() else 2,
            callbacks=[
                tf.keras.callbacks.LearningRateScheduler(scheduler.schedule, verbose=1),
                tf.keras.callbacks.TensorBoard(log_dir=args.model_dir, update_freq=10),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(args.model_dir, 'checkpoint'),
                    verbose=1,
                ),
            ]
        )
        model.save_weights(os.path.join(args.model_dir, 'checkpoint'), save_format='tf')
        print(f'Checkpoint is saved to {args.model_dir}.')

    if mode == 'export' or args.export:
        model.load_weights(os.path.join(args.model_dir, 'checkpoint'))
        model.inputs = None
        model._set_inputs(np.zeros([1, 256, 128, 3], dtype=np.float))
        model.outputs = model.outputs[::-1]
        model.output_names = model.output_names[::-1]
        model.save(args.output, save_format='tf')
        print(f'Saved to {args.output}')

    # config_proto = tf.compat.v1.ConfigProto(log_device_placement=True)
    # config = tf.estimator.RunConfig(
    #     model_dir=args.model_dir,
    #     save_summary_steps=100,
    #     keep_checkpoint_max=5,
    #     log_step_count_steps=10,
    #     session_config=config_proto,
    # )
    # estimator_model = tf.keras.estimator.model_to_estimator(
    #     keras_model=model,
    #     model_dir=args.model_dir,
    #     config=config,
    #     checkpoint_format='saver',
    # )
    #
    # if mode == 'train':
    #     estimator_model.train(
    #         input_fn=dataset.get_input_fn,
    #         steps=args.steps,
    #     )
    # elif mode == 'export':
    #     saved_path = estimator_model.export_saved_model(
    #         args.model_dir,
    #     )
    #     print(f'Saved to {saved_path}.')


if __name__ == '__main__':
    main()