import jax.numpy as jnp
import tensorflow_datasets as tfds

def load_data(name, batch_size=-1):
  """Load train and test datasets into memory."""
  ds_builder = tfds.builder(name)
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=batch_size))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=batch_size))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  
  return train_ds, test_ds