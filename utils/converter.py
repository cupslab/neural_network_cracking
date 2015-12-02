#!/usr/bin/env python3

import sys
import argparse
import csv
import os.path

class PgsFile(object):
    def __init__(self, ifile, gcolumn = 2, rounding = False):
        self.ifile = ifile
        self.gcolumn = gcolumn

    def as_tuples(self):
        def get_value(row):
            value = row[self.gcolumn - 1]
            try:
                value = float(value)
                return str(round(value))
            except ValueError:
                return value
        answer = [(row[0], get_value(row)) for row in csv.reader(
            self.ifile, delimiter = '\t', quotechar=None)]
        self.ifile.close()
        return answer

    def as_dict(self):
        return dict(self.as_tuples())

    @staticmethod
    def max_in_dict(adict):
        def try_int(k):
            value = adict[k]
            try:
                return int(value)
            except ValueError:
                return round(float(value))
        return max(map(try_int, adict))

USER_COLUMN = 'no_user'
PROBABILITY_COLUMN = '0x0.1p-1'
MASK_COLUMN = 'WRGOMI'
OFILE_SUFFIX = ''
LOOKUP_PREFIX = 'lookupresults.'
TOTAL_COUNT_PREFIX = 'totalcounts.'
TOTAL_COUNT_FORMAT = '%s:Total count\t%d\n'
TOTAL_COUNT_STDIN = '(standard input)'

class ConditionFiles(object):
    def __init__(self, file_list, names = None):
        self.file_list = file_list
        self.names = list(
            map(str, range(len(file_list)))) if names is None else names
        assert len(self.names) == len(self.file_list)

    def as_dict_list(self):
        answer = {}
        for afile in range(len(self.file_list)):
            answer[self.names[afile]] = self.file_list[afile]
        return answer

    def as_weir_tuple(self, index, pgs_dict):
        answer = []
        for pwd in self.file_list[index]:
            if pwd in pgs_dict:
                guess = pgs_dict[pwd]
            else:
                guess = '-1000'
                sys.stderr.write(
                    'Error, "%s" from "%s" is not present in guess results\n' %
                    (pwd, self.names[index]))
            answer.append((USER_COLUMN, self.names[index], pwd,
                           PROBABILITY_COLUMN, '', guess, MASK_COLUMN))
        return answer

    def weir_file_name(self, index, odir):
        def prefix(pre):
            return os.path.join(
                odir, pre + self.names[index] + OFILE_SUFFIX)
        return (prefix(LOOKUP_PREFIX), prefix(TOTAL_COUNT_PREFIX))

    def write_weir_file(self, index, pgs_dict, ofile):
        writer = csv.writer(ofile, delimiter = '\t', lineterminator = '\n',
                            quotechar = None)
        for row in self.as_weir_tuple(index, pgs_dict):
            writer.writerow(row)

    def write_weir_totals(self, pgs_dict, ofile, index = -1):
        # TODO: Find the real max guess number
        ofile.write(TOTAL_COUNT_FORMAT % (
            self.names[index] if index != -1 else TOTAL_COUNT_STDIN,
            PgsFile.max_in_dict(pgs_dict)))

    def write_weir_files(self, pgs_dict, odir):
        for idx in range(len(self.file_list)):
            lookup_name, total_name = self.weir_file_name(idx, odir)
            with open(lookup_name, 'w') as lookup_file:
                self.write_weir_file(idx, pgs_dict, lookup_file)
            with open(total_name, 'w') as total_file:
                self.write_weir_totals(pgs_dict, total_file, idx)

class ConditionNames(object):
    def __init__(self, files, name_file = None):
        self.files = files
        self.name_file = name_file

    def get_names(self):
        if self.name_file is not None:
            name_map = dict([(row['file'], row['name'])
                             for row in csv.DictReader(self.name_file)])
            self.name_file.close()
            return list(map(lambda x: name_map[x], self.names()))
        return self.names()

class ConditionNamesPlain(ConditionNames):
    def names(self):
        return list(map(
            lambda x: os.path.basename(x.name).replace('.', ''), self.files))

    def file_list(self):
        answer = []
        for afile in self.files:
            answer.append([row.strip('\n') for row in afile])
            afile.close()
        return answer

class ConditionNamesCsv(ConditionNames):
    def __init__(self, files, name_file = None):
        if len(files) != 1:
            raise Exception('Expecting exactly one password file')
        super().__init__(files, name_file)
        self.data = [(row[0], row[1]) for row in csv.reader(files[0])]
        self._names = sorted(list(dict(self.data).keys()))
        files[0].close()

    def names(self):
        return self._names

    def file_list(self):
        answer = []
        for name in self._names:
            answer.append(list(map(
                lambda x: x[1], filter(lambda x: x[0] == name, self.data))))
        return answer

def main_args(ifile, condition_files, condition_names_file,
              odir, csv_format = False, gcolumn = 2, rounding = False):
    if csv_format:
        c = ConditionNamesCsv(condition_files, condition_names_file)
    else:
        c = ConditionNamesPlain(condition_files, condition_names_file)
    ConditionFiles(c.file_list(), c.get_names()).write_weir_files(
        PgsFile(ifile, gcolumn, rounding).as_dict(), odir)

def main(args):
    main_args(args.ifile, args.condition_files, args.condition_names_file,
              args.odir, args.condition_files_csv_format, args.guess_column)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="""Convert PGS output into files to feed into graphs.

The PGS output gives a tsv with 2 columns (password, guess #). We want to add a
mapping of passwords to conditions and then output multiple files for input into
graphing code. This will output two files per condition (lookupresults and
totalcounts). To run the unit tests, run

$ python3 -m unittest converter-unittests""")
    parser.add_argument('ifile', type = argparse.FileType('rU'), help =
                        """Input file from PGS. Should be a TSV with two
columns, password and guess number. Default is stdin.""")
    parser.add_argument('--odir', default='.',
                        help="""Output directory, default is current
directory. This may overwrite files here. """)
    parser.add_argument('condition_files', nargs='+',
                        type = argparse.FileType('rU'),
                        help="""Condition files. A list of passwords in each
condition""")
    parser.add_argument('--condition-names-file', help="""File for condition
names, it is a csv with columns: file, name. """,
                        type = argparse.FileType('rU'))
    parser.add_argument('--condition-files-csv-format', action='store_true',
                        help = """If true, then the condition_files arguments
are in csv format with two columns, first column is the condition name, second
column is the password. """)
    parser.add_argument('--guess-column', type=int, default=2,
                        help=('Column number of the guess column in the input'
                              ' file. Default is 2. This column is 1-indexed'))
    parser.add_argument('--rounding', action='store_true',
                        help='Round floats')
    main(parser.parse_args())
