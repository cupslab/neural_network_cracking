import unittest
import unittest.mock
import io
import tempfile
import shutil

import converter

class PgsFileTest(unittest.TestCase):
    def test_as_tuples(self):
        input_str = io.StringIO("""testpwd	-5
asdfasdf	49949""")
        self.assertEqual([('testpwd', '-5'), ('asdfasdf', '49949')],
                         converter.PgsFile(input_str).as_tuples())

    def test_as_dict(self):
        input_str = io.StringIO("""testpwd	-5
asdfasdf	49949""")
        self.assertEqual({'testpwd' : '-5',
                          'asdfasdf' : '49949'},
                         converter.PgsFile(input_str).as_dict())

    def test_as_dict_quote_char(self):
        input_str = io.StringIO(""""testpwd"	-5
asdfasdf	49949""")
        self.assertEqual({'"testpwd"' : '-5',
                          'asdfasdf' : '49949'},
                         converter.PgsFile(input_str).as_dict())

    def test_max(self):
        input_str = io.StringIO("""testpwd	-5
asdfasdf	49949""")
        self.assertEqual(
            49949, converter.PgsFile.max_in_dict(
                converter.PgsFile(input_str).as_dict()))

    def test_gcolumn(self):
        input_str = io.StringIO(""""testpwd"	0.1	-5
asdfasdf	0.4	49949""")
        self.assertEqual({'"testpwd"' : '-5',
                          'asdfasdf' : '49949'},
                         converter.PgsFile(input_str, 3).as_dict())

class ConditionFiles(unittest.TestCase):
    def setUp(self):
        self.input_str_one = ['testpwd', 'anotherpwd']
        self.input_str_two = ['testpdw', 'fjsdjfjfj']
        self.input_str_spaces = [' testpdw', 'fjsdjfjfj']
        self.guesses = {
            'testpwd' : '1',
            'anotherpwd' : '2'
        }

    def test_as_dict_set(self):
        cond_files = converter.ConditionFiles(
            [self.input_str_one, self.input_str_two])
        self.assertEqual({ '0' : ['testpwd', 'anotherpwd'],
                           '1' : ['testpdw', 'fjsdjfjfj'] },
                         cond_files.as_dict_list())

    def test_as_dict_set_names(self):
        cond_files = converter.ConditionFiles(
            [self.input_str_one, self.input_str_two],
                                              ['name1', 'name2'])
        self.assertEqual({ 'name1' : ['testpwd', 'anotherpwd'],
                           'name2' : ['testpdw', 'fjsdjfjfj'] },
                         cond_files.as_dict_list())

    def test_as_dict_set_spaces(self):
        cond_files = converter.ConditionFiles(
            [self.input_str_spaces, self.input_str_two],
            ['name1', 'name2'])
        self.assertEqual({ 'name1' : [' testpdw', 'fjsdjfjfj'],
                           'name2' : ['testpdw', 'fjsdjfjfj'] },
                         cond_files.as_dict_list())

    def test_as_weir_tuple(self):
        cond_files = converter.ConditionFiles(
            [self.input_str_one, self.input_str_two], ['name1', 'name2'])
        self.assertEqual([(converter.USER_COLUMN, 'name1', 'testpwd',
                           converter.PROBABILITY_COLUMN, '', '1',
                           converter.MASK_COLUMN),
                          (converter.USER_COLUMN, 'name1', 'anotherpwd',
                           converter.PROBABILITY_COLUMN, '', '2',
                           converter.MASK_COLUMN)],
                         cond_files.as_weir_tuple(0, self.guesses))

    def test_weir_file_name(self):
        cond_files = converter.ConditionFiles(
            [self.input_str_one, self.input_str_two], ['name1', 'name2'])
        self.assertEqual(
            ('./lookupresults.name1', './totalcounts.name1'),
            cond_files.weir_file_name(0, './'))

    def test_weir_write_file(self):
        mock_output = io.StringIO()
        cond_files = converter.ConditionFiles(
            [self.input_str_one, self.input_str_two], ['name1', 'name2'])
        cond_files.write_weir_file(0, self.guesses, mock_output)
        self.assertEqual("""no_user	name1	testpwd	0x0.1p-1		1	WRGOMI
no_user	name1	anotherpwd	0x0.1p-1		2	WRGOMI
""", mock_output.getvalue())

    def test_write_weir_totals(self):
        mock_output = io.StringIO()
        cond_files = converter.ConditionFiles(
            [self.input_str_one, self.input_str_two], ['name1', 'name2'])
        cond_files.write_weir_totals(self.guesses, mock_output)
        self.assertEqual('(standard input):Total count	2\n',
                         mock_output.getvalue())

    def test_write_weir_totals_index(self):
        mock_output = io.StringIO()
        cond_files = converter.ConditionFiles(
            [self.input_str_one, self.input_str_two], ['name1', 'name2'])
        cond_files.write_weir_totals(self.guesses, mock_output, index = 0)
        self.assertEqual('name1:Total count	2\n',
                         mock_output.getvalue())

class ConditionNamesPlainTest(unittest.TestCase):
    def test_names_no_error(self):
        c = converter.ConditionNamesPlain([])
        c.get_names()
        c.file_list()

    def test_files(self):
        input_one = io.StringIO("asdf\nqwer")
        input_two = io.StringIO("jfjf\n poiu")
        input_one.name = 'stuff/name1.txt'
        input_two.name = 'name2.txt'
        c = converter.ConditionNamesPlain([input_one, input_two])
        self.assertEqual(['name1txt', 'name2txt'], c.get_names())
        self.assertEqual([['asdf', 'qwer'], ['jfjf', ' poiu']], c.file_list())

    def test_files_name_file(self):
        input_one = io.StringIO("asdf\nqwer")
        input_two = io.StringIO("jfjf\n poiu")
        input_one.name = 'stuff/name1.txt'
        input_two.name = 'name2.txt'
        input_names = io.StringIO("file,name\nname1txt,Cond1\nname2txt,Cond2\n")
        c = converter.ConditionNamesPlain([input_one, input_two], input_names)
        self.assertEqual(['Cond1', 'Cond2'], c.get_names())
        self.assertEqual([['asdf', 'qwer'], ['jfjf', ' poiu']], c.file_list())

class ConditionNamesCsvTest(unittest.TestCase):
    def test_names_no_error(self):
        input_one = io.StringIO('')
        input_one.close = unittest.mock.MagicMock()
        input_one.name = ''
        c = converter.ConditionNamesCsv([input_one])
        c.get_names()
        c.file_list()

    def test_names_file(self):
        input_one = io.StringIO("cond1,asdf\ncond2,qwer\ncond1,fjfj")
        input_one.close = unittest.mock.MagicMock()
        input_one.name = ''
        c = converter.ConditionNamesCsv([input_one])
        self.assertEqual(['cond1', 'cond2'], c.get_names())
        self.assertEqual([['asdf', 'fjfj'], ['qwer']], c.file_list())

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_run_plain(self):
        ifile = io.StringIO("asdf\t1\nqwer\t2\njfjf\t3\n poiu\t4\n")
        input_one = io.StringIO("asdf\nqwer")
        input_two = io.StringIO("jfjf\n poiu")
        input_one.name = 'stuff/name1.txt'
        input_two.name = 'name2.txt'
        input_names = io.StringIO("file,name\nname1txt,Cond1\nname2txt,Cond2\n")
        converter.main_args(ifile, [input_one, input_two], input_names,
                            self.tempdir, False)

    def test_run_csv(self):
        ifile = io.StringIO("asdf\t1\nqwer\t2\nasfff\t3\n")
        input_one = io.StringIO("Cond1,asdf\nCond2,qwer\nCond1,asfff\n")
        input_one.name = 'stuff/name1.txt'
        converter.main_args(ifile, [input_one], None, self.tempdir, True)
