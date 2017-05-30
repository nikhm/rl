import sys

def pprint(txt):
    sys.stdout.write('\r')
    sys.stdout.write(txt)
    sys.stdout.flush()
