import string
import hashlib
import random

def md5(s):
    m = hashlib.md5()
    m.update(str(s).encode())
    return m.hexdigest()

def passwd():
    salt = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    print(salt)
    return salt