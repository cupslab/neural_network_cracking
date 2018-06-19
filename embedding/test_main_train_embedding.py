import os
import random
import shutil
import tempfile
import unittest

import pass_embedding as pe
import main_train_embedding as mte
import main_embedding_visualize as mev

default_config = """[embedding]
num_train_epochs = 1
logging_freq = 1
password_batch = 32
batch_size = 4"""

class TestMain(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _temp_path(self, name):
        return os.path.join(self.test_dir, name)

    def test_make_parser(self):
        mte.make_parser()

    def _run_training(self):
        ifile_name = self._temp_path('input_file')
        ofile_name = self._temp_path('output_file')
        config_name = self._temp_path('config_file')
        with open(ifile_name, 'w') as ifile_data:
            for _ in range(100):
                ifile_data.write(''.join([
                  random.choice(pe.default_alphabet)
                  for _ in range(random.randrange(1, 20))]))

        with open(config_name, 'w') as config_data:
            config_data.write(default_config)

        args = mte.make_parser().parse_args([
          '--input-file', ifile_name,
          '--output-file', ofile_name,
          '--config', config_name])
        mte.main(args)
        return ofile_name, config_name


    def test_train(self):
        self._run_training()

    def test_visualize_analogy(self):
        ofile_name, config_name = self._run_training()
        parser = mev.make_parser()
        mev.main(
          parser.parse_args([config_name, ofile_name, '--analogy', "ab AB cd"]))
