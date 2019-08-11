import tarfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename, 'r:bz2')
    tar.extractall('data')
    tar.close()


if __name__ == "__main__":
    extract('data/LJSpeech-1.1.tar.bz2')
