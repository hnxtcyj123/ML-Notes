# 层初始化错误

"error": "Could not find variable batch_normalization/moving_mean. This could mean that the variable has been deleted. In TF1, it can also mean the variable is uninitialized. Debug info: container=localhost, status=Not found: Container localhost does not exist. (Could not find resource: localhost/batch_normalization/moving_mean)\n\t [[{{node model/sequential/batch_normalization/batchnorm/ReadVariableOp_1}}]]"

此错误为模型参数未加载所致，具体原因如下：

首先，tf模型目录结构如下（tf2.10）：

├── assets

├── keras_metadata.pb

├── saved_model.pb

└── variables

    ├── variables.data-00000-of-00001

    └── variables.index

在上传模型压缩包到服务端后，会解压到服务加载模型的路径，当解压出saved_model.pb时服务就会检测到并开始加载切换模型，如果解压顺序按照saved_model.pb、variables/variables.data-00000-of-00001、variables/variables.index，由于variables/variables.data-00000-of-00001比较大，此时模型已检测到saved_model.pb开始加载模型，但variables/variables.data-00000-of-00001还未解压完成，tf-serving会提示未找到variables/variables.index

# 字符串切割维度错误

"error": "slice index 1 of dimension 0 out of bounds.\n\t [[{{function_node __inference__wrapped_model_421138}}{{node dnn_v1/user_list_layer/StringSplit_33/RaggedFromTensor/strided_slice}}]]"

此错误发生在使用tf.strings.split时，在tf模型中为了保证线上线下的一致性，应该尽可能把预处理部分也放在模型结构中，这样也可以保证输入的简单。在处理序列特征时，通常以字符串输入，在模型侧进行split得到序列（此操作会极大影响模型训练的效率）。

一开始定义的序列特征字符串预处理层如下：

```Python
# 输入shape=(None, 1)，第一维为batch_size
tf.keras.layers.Lambda(
    lambda x: tf.strings.split(x, sep=","), name="list_process_layer") # shape=(None, 1, None)
tf.keras.layers.Hashing(
    fc.vocab_size, mask_value=mask_token, name=name) # shape=(None, 1, None)
tf.keras.layers.Embedding(
    fc.vocab_size, embedd_dim, mask_zero=True, name=name) # shape=(None, 1, None, emb_dim)
tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1)) # shape=(None, None, emb_dim)
tf.keras.layers.GlobalAveragePooling1D(
    keepdims=True, name="mean_pooling") # shape=(None, 1, emb_dim)
```

预处理层在训练时没有问题，在serving时会提示上述错误，那么尝试在输入部分进行强制reshape，此时输入少了一维，把tf.squeeze操作去掉：

```Python
tf.keras.layers.Lambda(lambda x: tf.strings.split(tf.reshape(x, shape=(-1,)), sep="$,"), 
                                                  name="list_process_layer") # shape=(None, None)
tf.keras.layers.Hashing(fc.vocab_size, mask_value=mask_token, name=name) # shape=(None, None)
tf.keras.layers.Embedding(fc.vocab_size, embedd_dim, mask_zero=True, name=name) # shape=(None, None, emb_dim)
tf.keras.layers.GlobalAveragePooling1D(keepdims=True, name="mean_pooling") # shape=(None, 1, emb_dim)
```

此时上述问题解决，但会报concat维度不一致的错误，序列特征在embedding和pooling之后需要和其他embedding进行concat操作，此时序列embedding为2D tensor，而其他embedding为3D tensor。同样在训练和预测时都没有问题（序列embedding和其他embedding一样为3D tensor），在serving时报错。怀疑是RaggedTensor在serving时有不一样的默认处理（比如训练模型时在RaggedTensor生成时会默认增加一维，在serving时不会？因为此时pooling后embedding维度少了一维，训练时打印维度都是对的上的，serving时不方便定位），那么在pooling后再加上强制reshape：

```Python
tf.keras.layers.Lambda(lambda x: tf.strings.split(tf.reshape(x, shape=(-1,)), sep=","), 
                                                  name="list_process_layer") # shape=(None, None)
tf.keras.layers.Hashing(fc.vocab_size, mask_value=mask_token, name=name) # shape=(None, None)
tf.keras.layers.Embedding(fc.vocab_size, embedd_dim, mask_zero=True, name=name) # shape=(None, None, emb_dim)
tf.keras.layers.GlobalAveragePooling1D(keepdims=True, name="mean_pooling") # shape=(None, 1, emb_dim)
tf.keras.layers.Lambda(lambda x: tf.reshape(x, shape=(-1, 1, embedd_dim))) # shape=(None, 1, emb_dim)
```

此时serving能正常进行预测，问题解决。

# 服务热切换模型延迟大

服务热切换模型时会有秒级别的延迟，对于推荐排序模型这种需要频繁更新的模型是不可忍受的。热启动是指在模型启动时加载一些预先准备好的请求，以便让模型预热，提高后续的响应速度。

SavedModel 预热支持回归、分类、多重推理和 预测。要在加载时触发模型预热，需要附加预热数据文件 在 SavedModel 目录的 assets.extra 子文件夹下。

模型预热正常工作的要求：

* 预热文件名：'tf_serving_warmup_requests'
* 文件位置： assets.extra/
* 文件格式： [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details) 以每条记录作为[预测日志](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto#:%7E:text=message-,PredictionLog,-%7B).
* 预热记录数 <= 1000。

生成预热数据的代码片段：

```Python
def prepare_warmup_files(args, paths):
    """Generate TFRecords for warming up."""
    def build_request_inputs(predict_request, sample, feature_columns, dtype):
        for fc in feature_columns:
            predict_request.inputs[fc.fname].CopyFrom(tf.make_tensor_proto(sample[fc.fname], dtype=dtype))
        return predict_request
  
    # prepare data
    dataset: tf.data.Dataset = load_dataset(filenames, batch_size=1, shuffle_size=100000)

    system_command(f"mkdir -p {paths.warmup_file_path}")

    # generate warmup requests
    NUM_RECORDS = 100
    with tf.io.TFRecordWriter(os.path.join(warmup_file_path, "tf_serving_warmup_requests")) as writer:
        for i, sample in tqdm(enumerate(dataset)):
            if i >= NUM_RECORDS:
                break
            predict_request = predict_pb2.PredictRequest()
            predict_request.model_spec.name = args.model_name
            predict_request.model_spec.signature_name = "serving_default"
            predict_request = build_request_inputs(predict_request, sample, sparse_fnames, tf.string)
            predict_request = build_request_inputs(predict_request, sample, dense_fnames, tf.float32)
            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(request=predict_request))
            writer.write(log.SerializeToString())
```

# tf-serving模型热更新内存泄漏

模型版本热更新时，某个版本已经卸载，但内存没有释放。经过反复google，发现把 malloc 换成 jemalloc 可以解决，估计是malloc本身存在内存泄漏问题，这个bug在github上被反复提issue，因为每个人的环境不同，官方也没有复现，最后不了了之，最后还是靠广大群众自己研究出来的解决方案，官方竟然也不去修复？魔改 Dockerfile 后重新构建即可，加上：

```Shell
RUN apt-get update && apt-get install -y libjemalloc-dev
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.1
```
