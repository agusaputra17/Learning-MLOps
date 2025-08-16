from kerastuner.tuners import RandomSearch
import tensorflow as tf
import tensorflow_transform as tft 
from tensorflow.keras import layers
import keras_tuner as kt
from keras_tuner import Hyperband
import os  
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult


LABEL_KEY = "label"
FEATURE_KEY = "text"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=64) -> tf.data.Dataset:
    """Get post_transform feature & create batches of data"""
    
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=1,
        label_key=transformed_name(LABEL_KEY)
    )
    
    dataset = dataset.repeat(num_epochs)
    
    return dataset

# Vocabulary size and number of words in a sequence.
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

@tf.autograph.experimental.do_not_convert
# Fungsi untuk adaptasi TextVectorization
def adapt_vectorize_layer(dataset):
    text_data = dataset.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
    vectorize_layer.adapt(text_data)
    
# Fungsi Model dengan Hyperparameter Tuning
def model_builder(hp):
    embedding_dim = hp.Int("embedding_dim", min_value=16, max_value=128, step=16)
    dense_units_1 = hp.Int("dense_units_1", min_value=32, max_value=128, step=32)
    dense_units_2 = hp.Int("dense_units_2", min_value=16, max_value=64, step=16)
    learning_rate = hp.Choice("learning_rate", values=[0.001, 0.005, 0.01])

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units_1, activation='relu')(x)
    x = layers.Dense(dense_units_2, activation="relu")(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    return model

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            mode='min',
                                            patience=2, 
                                            restore_best_weights=True)

def tuner_fn(fn_args):
    """Build the tuner using the KerasTuner API."""
    # Memuat training dan validation dataset yang telah di-preprocessing
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Pastikan input_fn menggunakan .repeat()
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=5, batch_size=128)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=5, batch_size=128)

    # Adaptasi vectorize_layer sebelum digunakan
    adapt_vectorize_layer(train_set)
    
    # Definisikan strategi hyperparameter tuning
    tuner = RandomSearch(
        model_builder,
        objective='val_binary_accuracy',
        # max_epochs=5,
        max_trials = 5,
        # factor=3,
        directory=fn_args.working_dir,
        project_name='imdb_tuner',
        
    )

    return TunerFnResult (
        tuner=tuner,
        fit_kwargs={
            "callbacks": [callback],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
