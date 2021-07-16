import os
import sys
import time
import datetime
from pprint import pprint as pp
from pprint import pformat as pf
import numpy as np
from sklearn.metrics import classification_report

from data import read_test_xlsx


def main():
    te_texts = read_test_xlsx()
    te_labels = None

if __name__ == '__main__':
    print(classification_report())
