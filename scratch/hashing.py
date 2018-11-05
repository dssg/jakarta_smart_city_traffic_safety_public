import hashlib
import os

# directory_str = 'data2/'
directory_str = '/to_raw/videos2/'
output_file = 'hashes2.csv'
directory = os.fsencode(directory_str)

f = open(output_file, 'w')


def hash_bytestr_iter(bytesiter, hasher, ashexstr=False):
    for block in bytesiter:
        hasher.update(block)
    return (hasher.hexdigest() if ashexstr else hasher.digest())


def file_as_blockiter(afile, blocksize=65536):
    with afile:
        block = afile.read(blocksize)
        while len(block) > 0:
            yield block
            block = afile.read(blocksize)


for i, filename in enumerate(os.listdir(directory)):
    filename = filename.decode('utf-8')
    print(i, filename)
    f.write(filename)
    f.write(', ')
    f.write(hash_bytestr_iter(file_as_blockiter(open(directory_str + filename, 'rb')),
                              hashlib.md5(), True))
    f.write('\n')

f.close()
