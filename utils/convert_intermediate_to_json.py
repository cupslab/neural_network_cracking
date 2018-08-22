#!/usr/bin/env python3
from sqlitedict import SqliteDict
import argparse
import os.path
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert intermediate data sqlite files to json format")
    parser.add_argument("infile")
    parser.add_argument("outfile")
    args = parser.parse_args()
    if not os.path.isfile(args.infile):
        raise FileNotFoundError("The intermediate sqlite file does not exist")
    json_dict ={}
    sqldict = SqliteDict(filename=args.infile)
    for key in sqldict:
        json_dict[key] = sqldict[key]
    with open(args.outfile, "w") as outfile:
        json.dump(json_dict, outfile)
