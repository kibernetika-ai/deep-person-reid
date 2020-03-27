import argparse
import glob
import re
import os

import tensorflow as tf

from torchreid.models import osnet_tf


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--model-dir', default='train')
    parser.add_argument('--data-dir')
    parser.add_argument('--dataset-name')
    parser.add_argument('--steps', type=int, default=1000)

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


def main():
    args = parse_args()
    dataset = Market1501Dataset(args.data_dir, args.mode, args.batch_size)
    model = osnet_tf.osnet_x0_25(num_classes=dataset.num_classes())

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1,
        ),
        metrics=['accuracy']
    )
    mode = args.mode

    if mode == 'train':
        history = model.fit(
            x=dataset.get_input_fn(),
            # batch_size=args.batch_size,
            epochs=1,
        )
        print(history)

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