import sys, os

for path in sys.path:
    if os.path.isdir(path + '/astroabc'):
        os.system('rm -r ' + path + '/astroabc')
        print('rm -r ' + path + '/astroabc')
    if os.path.isdir(path + '/bgmfast'):
        print('cp -r ' + path + '/bgmfast/astroabc ' + path)
        os.system('cp -r ' + path + '/bgmfast/astroabc ' + path)
