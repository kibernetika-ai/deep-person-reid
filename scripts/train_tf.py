import argparse
import glob
import re
import os

import tensorflow as tf

from torchreid.models import osnet_tf
from torchreid.utils import utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--mode', default='train')
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
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid))
            paths.append(img_path)
            pids.append(pid)

        self.paths = paths
        self.pids = pids

    def get_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.paths, self.pids))
        return dataset.map(preprocess).batch(self.batch_size)

    def num_classes(self):
        return len(set(self.pids))


def preprocess(image_path, label):
    raw_content = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw_content, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image to the desired size.
    image = tf.image.resize(image, [256, 128])
    image = (image / 127.5) - 1
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
        zeros = tf.zeros([self.batch_size, self.num_classes])
        # targets = utils.scatter_numpy(zeros.numpy(), 1, tf.expand_dims(y_true, 1).numpy(), 1)
        targets = tf.scatter_nd(tf.expand_dims(y_true, 1), 1, [self.batch_size, self.num_classes])
        # if self.use_gpu:
        #     targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return tf.reduce_sum(tf.reduce_mean(-targets * log_probs, axis=0))


def main():
    args = parse_args()
    dataset = Market1501Dataset(args.data_dir, args.mode, args.batch_size)
    model = osnet_tf.osnet_x0_25(num_classes=dataset.num_classes())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        # loss=CrossEntropyLoss(num_classes=dataset.num_classes(), batch_size=args.batch_size),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    mode = args.mode

    if mode == 'train':
        history = model.fit(
            x=dataset.get_input_fn(),
            # batch_size=args.batch_size,
            epochs=args.epochs,
        )
        print(history)
        model.save_weights(os.path.join(args.model_dir, 'checkpoint'), save_format='tf')
    elif mode == 'export':
        model.load_weights(os.path.join(args.model_dir, 'checkpoint'))
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