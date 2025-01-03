import functools
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_function


BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def Activation(inputs, activation='relu'):
  """Only supports ReLU and SiLU/Swish."""
  assert activation in ['relu', 'silu']
  if activation == 'relu':
    return tf.nn.relu(inputs)
  else:
    return tf.nn.swish(inputs)


def BNReLU(
    inputs, is_training, nonlinearity=True,
    init_zero=False, activation='relu'):
  """Performs a batch normalization followed by a ReLU."""
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=3,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training,
        fused=True,
        gamma_initializer=gamma_initializer)

  if nonlinearity:
    inputs = Activation(inputs, activation=activation)

  return inputs


def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size."""
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(
      inputs, [[0, 0], [pad_beg, pad_end], pad_beg, pad_end], [0, 0]])
  return padded_inputs


def Conv2D(inputs, *, filters, kernel_size, strides=1):
  """Strided 2-D convolution with explicit padding."""
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(
          scale=2., mode='fan_in', distribution='untruncated_normal'))


# Functions `rel_to_abs`, `relative_logits_1d`, `relative_logits`
# and `relpos_self_attention` are fully based on
# https://github.com/tensorflow/tensor2tensor/blob/21dba2c1bdcc7ab582a2bfd8c0885c217963bb4f/tensor2tensor/layers/common_attention.py#L2225.


def rel_to_abs(x):
  """
  Converts relative indexing to absolute.
  Input: [bs, heads, length, 2*length - 1]
  Output: [bs, heads, length, length]
  """
  bs, heads, length, _ = x.shape
  col_pad = tf.zeros((bs, heads, length, 1), dtype=x.dtype)
  x = tf.concat([x, col_pad], axis=3)
  flat_x = tf.reshape(x, [bs, heads, -1])
  flat_pad = tf.zeros((bs, heads, length-1), dtype=x.dtype)
  flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
  final_x = tf.reshape(
    flat_x_padded, [bs, heads, length+1, 2*length-1])
  final_x = final_x[:, :, :length, length-1:]
  return final_x


def relative_logits_1d(*, q, rel_k, transpose_mask):
  """
  Compute relative logits along one dimenion.
  `q`: [bs, heads, height, width, dim]
  `rel_k`: [2*width - 1, dim]
  """
  bs, heads, h, w, dim = q.shape
  rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
  rel_logits = tf.reshape(rel_logits, [-1, heads * h, w, 2*w-1])
  rel_logits = rel_to_abs(rel_logits)
  rel_logits = tf.reshape(rel_logits, [-1, heads, h, w, w])
  rel_logits = tf.expand_dims(rel_logits, axis=3)
  rel_logits = tf.tile(rel_logits, [1, 1, 1, h, 1, 1])
  rel_logits = tf.transpose(rel_logits, transpose_mask)
  return rel_logits


def relative_logits(q):
  """Compute relative position enc logits."""
  with tf.variable_scope('relative', reuse=tf.AUTO_REUSE):
    bs, heads, h, w, dim = q.shape
    int_dim = dim.value
    # Note: below, we passed stddev arg as mean for the initializer.
    # Providing code as is, with this small error.
    # right way: normal_initializer(stddev=int_dim**-0.5)

    # Relative logits in width dimension.
    rel_emb_w = tf.get_variable(
        'r_width', shape=(2*w - 1, dim),
        dtype=q.dtype,
        initializer=tf.random_normal_initializer(int_dim**-0.5))
    rel_logits_w = relative_logits_1d(
      q=q, rel_k=rel_emb_w,
      transpose_mask=[0, 1, 2, 4, 3, 5])

    # Relative logits in height dimension.
    rel_emb_h = tf.get_variable(
        'r_height', shape=(2*h - 1, dim),
        dtype=q.dtype,
        initializer=tf.random_normal_initializer(int_dim**-0.5))
    rel_logits_h = relative_logits_1d(
        q=tf.transpose(q, [0, 1, 3, 2, 4]),
        rel_k=rel_emb_h,
        transpose_mask=[0, 1, 4, 2, 5, 3])
    return rel_logits_h + rel_logits_w


def relpos_self_attention(
  *, q, k, v, relative=True, fold_heads=False):
  """2D self-attention with rel-pos. Add option to fold heads."""
  bs, heads, h, w, dim = q.shape
  int_dim = dim.value
  q = q * (dim ** -0.5) # scaled dot-product
  logits = tf.einsum('bhHWd,bhPQd->bhHWPQ', q, k)
  if relative:
    logits += relative_logits(q)
  weights = tf.reshape(logits, [-1, heads, h, w, h * w])
  weights = tf.nn.softmax(weights)
  weights = tf.reshape(weights, [-1, heads, h, w, h, w])
  attn_out = tf.einsum('bhHWPQ,bhPQd->bHWhd', weights, v)
  if fold_heads:
    attn_out = tf.reshape(attn_out, [-1, h, w, heads * dim])
  return attn_out

def absolute_logits(q):
  """Compute absolute position enc logits."""
  with tf.variable_scope('absolute', reuse=tf.AUTO_REUSE):
    emb_w = tf.get_variable(
        'r_width', shape=(W, dkh),
        dtype=q.dtype,
        initializer=tf.random_normal_initializer(dkh**-0.5))
    emb_h = tf.get_variable(
        'r_height', shape=(H, dkh),
        dtype=q.dtype,
        initializer=tf.random_normal_initializer(dkh**-0.5))
    emb_h = emb_h[:, None, :]
    emb_w = emb_w[None, :, :]
    emb = emb_h + emb_w
    abs_logits = tf.einsum('bhxyd,pqd->bhxypq', q, emb)
    return abs_logits


def abspos_self_attention(*, q, k, v, absolue=True, fold_heads=False):
  """2D self-attention with abs-pos. Add option to fold heads."""
  bs, heads, h, w, dim = q.shape
  int_dim = dim.value
  q = q * (dim ** -0.5) # scaled dot-product
  logits = tf.einsum('bhHWd,bhPQd->bhHWPQ', q, k)
  abs_logits = absolute_logits(q)
  if absolute:
    logits += abs_logits
  weights = tf.reshape(logits, [-1, heads, h, w, h * w])
  weights = tf.nn.softmax(weights)
  weights = tf.reshape(weights, [-1, heads, h, w, h, w])
  attn_out = tf.einsum('bhHWPQ,bhPQd->bHWhd', weights, v)
  if fold_heads:
    attn_out = tf.reshape(attn_out, [-1, h, w, heads * dim])
  return attn_out


def group_pointwise(
  featuremap, proj_factor=1, name='grouppoint',
  heads=4, target_dimension=None):
  """1x1 conv with heads."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    in_channels = featuremap.shape[-1]
    if target_dimension is not None:
      proj_channels = target_dimension // proj_factor
    else:
      proj_channels = in_channels // proj_factor
    w = tf.get_variable(
        'w',
        [in_channels, heads, proj_channels // heads],
        dtype=featuremap.dtype,
        initializer=tf.random_normal_initializer(stddev=0.01))
    out = tf.einsum('bHWD,Dhd->bhHWd', featuremap, w)
    return out


def MHSA(featuremap, pos_enc_type='relative', use_pos=True):
  """Multi-Head Self-Attention."""
  q = group_pointwise(
      featuremap, proj_factor=1, name='q_proj', heads=heads,
      target_dimension=bottleneck_dimension)
  k = group_pointwise(
      featuremap, proj_factor=1, name='k_proj', heads=heads,
      target_dimension=bottleneck_dimension)
  v = group_pointwise(
      featuremap, proj_factor=1, name='v_proj', heads=heads,
      target_dimension=bottleneck_dimension)
  assert pos_enc_type in ['relative', 'absolute']
  if pos_enc_type == 'relative':
    o = relpos_self_attention(
        q=q, k=k, v=v, relative=use_pos, fold_heads=True)
  else:
    o = abspos_self_attention(
        q=q, k=k, v=v, absolute=use_pos, fold_heads=True)
  return o


def BoT_Block(
    featuremap, is_training=False,
    heads=4, proj_factor=4,
    activation='relu',
    pos_enc_type='relative',
    name='all2all', strides=1,
    target_dimension=2048):
  """Bottleneck Transformer (BoT) Block."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    shortcut = featuremap
    in_dimension = featuremap.shape[-1]
    if strides != 1 or in_dimension != target_dimension:
      shortcut = Conv2D(
        shortcut, filters=target_dimension, kernel_size=1, strides=strides)
      shortcut = BNReLU(
          shortcut, is_training, activation=activation, nonlinearity=True)

    bottleneck_dimension = target_dimension // proj_factor
    featuremap = Conv2D(
        featuremap, filters=bottleneck_dimension, kernel_size=1, strides=1)
    featuremap = BNReLU(
        featuremap, is_training, activation=activation, nonlinearity=True)

    featuremap = MHSA(featuremap, pos_enc_type=pos_enc_type)
    if strides != 1:
      assert strides == 2
      featuremap = tf.keras.layers.AveragePooling2D(
          pool_size=(2, 2), strides=(2, 2), padding='same')(featuremap)
    featuremap = BNReLU(
        featuremap, is_training, activation=activation, nonlinearity=True)

    featuremap= Conv2D(
        featuremap, filters=target_dimension,
        kernel_size=1, strides=1)
    featuremap = BNReLU(
        featuremap, is_training, nonlinearity=False, init_zero=True)

    return Activation(shortcut + featuremap, activation=activation)


def BoT_Stack(
    featuremap, *,
    blocks_so_far,
    total_blocks,
    is_training=False,
    heads=4, proj_factor=4,
    activation='relu',
    pos_enc_type='relative',
    name='all2all_stack',
    strides=2, num_layers=3,
    target_dimension=2048):
  """c5 Blockgroup of BoT Blocks."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    for i in range(num_layers):
      featuremap = BoT_Block(
          featuremap,
          is_training=is_training,
          heads=heads,
          proj_factor=proj_factor,
          activation=activation,
          pos_enc_type=pos_enc_type,
          strides=strides if i == 0 else 1,
          target_dimension=target_dimension,
          name='all2all_layer_{}'.format(i))
    return featuremap