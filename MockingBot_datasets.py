'''
Prepares datasets for training.
'''
import matplotlib.pyplot as plt
import os
import pathlib
import tensorflow as tf
import random
import wave


class DatasetManager:
    '''
    Holds a collection of static and dynamic datasets.
    '''

    __datasets = {
        # name: {
        #   source: Path | callable,
        #   build: None | tf.Dataset
        # }
    }

    def __init__(self, dataset_sources) -> None:
        '''
        Initializes a `DatasetManager`.
        '''
        self.register(dataset_sources)


    def build(self, dataset_name) -> tf.data.Dataset:
        '''
        Builds a dataset (loads it into memory as a `tf.data.Dataset`).

        dataset_name: The dataset to build.

        Return:
        '''
        source = self.__datasets[dataset_name]['source']

        if callable(source):
            dataset = source()

        elif not callable(source):
            dataset = tf.data.Dataset.list_files(
                file_pattern=source.as_posix() + '/[!.]*.wav',
                shuffle=True,
                seed=0
            )

            filenames = [
                filename for filename in os.listdir(source)
                if not filename.startswith('.') and filename.endswith('.wav')
            ]

            test_filename = random.choice(filenames)

            test_file_path = source / test_filename

            with wave.open(str(test_file_path)) as wav:
                bit_depth = wav.getsampwidth() * 8

                sample_rate = wav.getframerate()

            dataset.sample_rate = sample_rate

            dataset.bit_depth = bit_depth

        self.__datasets[dataset_name]['build'] = dataset

        return dataset


    def discover(self, paths):
        '''
        Discovers and registers datasets in a path.

        paths: The path or paths to search.
        '''
        # Check if paths is iterable:
        if type(paths) in (str, pathlib.Path):
            paths = [paths]

        try:
            iter(paths)
        except TypeError:
            paths = [paths]

        for path in paths:
            source = pathlib.Path(path)

            if (source.exists()):
                for dataset_name in sorted(
                        os.listdir(source), key=str.lower):
                    if not dataset_name.startswith('.'):
                        print('DatasetManager.discover: Discovered:',
                            source / dataset_name)
                        self.register({dataset_name: source / dataset_name})
            else:
                print(
                    'DatasetManager: '
                    'WARNING: Failed to register dataset; '
                    'Path does not exist:', source
                )


    @property
    def datasets(self):
        '''
        The available datasets.
        '''
        return [str(key) for key in self.__datasets]


    def print_datasets(self):
        '''
        Searches data sources and prints available datasets.
        '''
        static_datasets = []

        dynamic_datasets = []

        for dataset_name, dataset_details in self.__datasets.items():
            source = dataset_details['source']
            if isinstance(source, pathlib.Path):
                if (source.exists()):
                    static_datasets.append(dataset_name)
                else:
                    print(
                        'DatasetManager: '
                        'WARNING: The following path chosen for static '
                        'datasets does not exist:', source
                    )
            elif callable(source):
                dynamic_datasets.append(dataset_name)
            else:
                raise TypeError(f'Invalid dataset source: {source}')

        print('Datasets:')

        print('- Static (stored on drive):')

        for dataset_name in static_datasets:
            print(f'\t{dataset_name}')

        print('- Dynamic (generated in memory):')

        for dataset_name in dynamic_datasets:
            print(f'\t{dataset_name}')


    def register(self, dataset_sources: dict):
        '''
        Registers a new static or dynamic dataset.

        dataset_source: A dict of the form {name: path | callable} where
            `name` is the name of the new dataset, `path` is a string or
            `pathlib.Path` to a static dataset, and `callable` is any callable
            which returns a TensorFlow `tf.data.Dataset`.
        '''
        valid_sources = {}

        for name, source in dataset_sources.items():
            if type(source) is str or isinstance(source, pathlib.Path):
                valid_sources.update(
                    {name: {'source': pathlib.Path(source), 'build': None}}
                )
            else:
                if not callable(source):
                    raise TypeError(
                        f'Invalid source: "{source}" of key "{name}" is not '
                        'a path or a callable'
                    )

                valid_sources.update({name: {'source': source, 'build': None}})

        self.__datasets.update(valid_sources)


    def view(self, dataset_name):
        '''
        Displays a random sample from a dataset.
        '''
        dataset_path = self.__datasets[dataset_name]['source']

        filenames = [
            filename for filename in os.listdir(dataset_path)
            if not filename.startswith('.') and filename.endswith('.wav')
        ]

        test_filename = random.choice(filenames)

        print('File name:', test_filename)

        test_file_path = dataset_path / test_filename

        with wave.open(test_file_path.as_posix()) as wav:
            bit_depth = wav.getsampwidth() * 8

            sample_rate = wav.getframerate()

        # print('Bit depth:', bit_depth)
        #
        # print('Sample rate:', sample_rate)

        # test_audio_tensor = tf.audio.decode_wav(
        #     contents=tf.io.read_file(test_file_path.as_posix()),
        #     desired_channels=1
        # ).audio
        #
        # print('Tensor dimensions:', test_audio_tensor.shape)
        #
        # plt.plot(test_audio_tensor)
        #
        # For Google Colab:
        # IPython.display.display(
        #     IPython.display.Audio(test_file_path, rate=sample_rate))

    def __getitem__(self, key):
        dataset_details = self.__datasets.get(key)

        if dataset_details is None:
            raise KeyError(f'No dataset found called: {key}')
        else:
            dataset = dataset_details.get('build')
            if dataset is None:
                return self.build(key)
            else:
                return dataset


    def test(self, dataset):
        '''
        Tests a dataset for invalid values.

        dataset: A `tf.data.Dataset`.
        '''
        '''
        # Test the dataset:
        for training_example in training_dataset.take(dataset_size):
            # Confirm there are no NaN's or inf's in the dataset:
            tf.debugging.assert_all_finite(
                training_example, str(training_example))

            # Confirm the values are normalized
            if not tf.experimental.numpy.isclose(
                tf.reduce_max(training_example).numpy(), 1):
            raise ValueError(f'Tensor is not normalized: {training_example}')
        '''
        return True
