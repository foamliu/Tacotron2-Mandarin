import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    zip = zipfile.ZipFile(filename, 'r')
    zip.extractall('data')
    zip.close()


if __name__ == "__main__":
    extract('data/BZNSYP.zip')
