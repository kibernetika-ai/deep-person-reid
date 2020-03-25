import argparse
from os import path
import shutil
import tempfile

import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import torch
import torch.onnx

import torchreid
from torchreid.utils import (
    check_isfile, collect_env_info,
    load_pretrained_weights, compute_model_complexity
)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model-name', type=str, default='', help='Model name'
    )
    parser.add_argument(
        '--weights', type=str, default='', help='Weights path'
    )
    parser.add_argument(
        '--output', type=str, default='output', help='Output path'
    )
    parser.add_argument(
        '--resolution', type=str, default='128x256', help='Resolution (WxH)'
    )

    args = parser.parse_args()
    width, height = [int(i) for i in args.resolution.split('x')]

    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    imagedata_kwargs = {
        'root': 'reid-data',
        'sources': ['market1501'],
        'targets': ['market1501'],
        'height': 256,
        'width': 128,
        'transforms': ['random_flip', 'color_jitter'],
        'norm_mean': [0.485, 0.456, 0.406],
        'norm_std': [0.229, 0.224, 0.225],
        'use_gpu': False,
        'split_id': 0,
        'combineall': False,
        'load_train_targets': False,
        'batch_size_train': 64,
        'batch_size_test': 300,
        'workers': 4,
        'num_instances': 4,
        'train_sampler': 'RandomSampler',
        'cuhk03_labeled': False,
        'cuhk03_classic_split': False,
        'market1501_500k': False
    }
    datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs)

    print('Building model: {}'.format(args.model_name))
    model = torchreid.models.build_model(
        name=args.model_name,
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True,
        use_gpu=False
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, height, width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if args.weights and check_isfile(args.weights):
        load_pretrained_weights(model, args.weights)
    _input = torch.Tensor(1, 3, height, width)
    inputs = (_input,)

    print('Converting PyTorch model to ONNX...')
    tmp = tempfile.mktemp(suffix='.onnx')
    torch.onnx._export(model, inputs, tmp, export_params=True)

    onnx_model = onnx.load(tmp)
    export_path = args.output

    onnx.checker.check_model(onnx_model)

    print('Prepare TF model...')
    tf_rep = prepare(onnx_model, strict=False)

    if path.exists(export_path):
        shutil.rmtree(export_path)

    with tf.Session() as persisted_sess:
        print("load graph")
        persisted_sess.graph.as_default()
        tf.import_graph_def(tf_rep.graph.as_graph_def(), name='')

        i_tensors = []
        o_tensors = []
        inputs = {}
        outputs = {}

        for i in tf_rep.inputs:
            t = persisted_sess.graph.get_tensor_by_name(
                tf_rep.tensor_dict[i].name
            )
            i_tensors.append(t)
            tensor_info = tf.saved_model.utils.build_tensor_info(t)
            inputs[t.name.split(':')[0].lower()] = tensor_info
            print(
                'input tensor [name=%s, type=%s, shape=%s]'
                % (t.name, t.dtype.name, t.shape.as_list())
            )
        print('')

        for i in tf_rep.outputs:
            t = persisted_sess.graph.get_tensor_by_name(
                tf_rep.tensor_dict[i].name
            )
            o_tensors.append(t)
            tensor_info = tf.saved_model.utils.build_tensor_info(t)
            outputs[t.name.split(':')[0]] = tensor_info
            print(
                'output tensor [name=%s, type=%s, shape=%s]'
                % (t.name, t.dtype.name, t.shape.as_list())
            )

        feed_dict = {}
        for i in i_tensors:
            feed_dict[i] = np.random.rand(*i.shape.as_list()).astype(i.dtype.name)

        print('test run:')
        res = persisted_sess.run(o_tensors, feed_dict=feed_dict)
        print(res)

        # print('INPUTS')
        # print(inputs)
        # print('OUTPUTS')
        # print(outputs)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        )
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            persisted_sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature
            })
        builder.save()
        print('Model saved to %s' % export_path)


if __name__ == '__main__':
    main()
