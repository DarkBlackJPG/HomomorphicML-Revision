from Crypto.Cipher import AES as OriginalAES
from Crypto.Util import Padding
from Crypto.Hash import SHA256

class AES:
    def __init__(self) -> None:
        pass
    
    def initialization(self, data: bytes, password: str, decrypt: bool):
        password_hash = SHA256.new()
        password_hash.update(bytes(password, 'utf-8'))
        password = password_hash.digest()
        padded_data = b''
        if not decrypt:
            padded_data = Padding.pad(data, OriginalAES.block_size)
        else:
            padded_data = data
        
        return {'password': password, 'data': padded_data}

    def encrypt(self, data: bytes, password: str, iv = None):
        prepared = self.initialization(data, password, decrypt=False)
        aes = None
        if iv is None:
            aes = OriginalAES.new(prepared['password'], OriginalAES.MODE_CBC)
        else:
            aes = OriginalAES.new(prepared['password'], OriginalAES.MODE_CBC, iv)

        encrypted = aes.encrypt(prepared['data'])
        iv = aes.iv
        return {'iv': iv, 'data': encrypted}

    def decrypt(self, data: bytes, password: str, iv):
        prepared = self.initialization(data, password, decrypt=True)
        aes = OriginalAES.new(prepared['password'], OriginalAES.MODE_CBC, iv)
        decrypted = aes.decrypt(prepared['data'])
        decrypted = Padding.unpad(decrypted, OriginalAES.block_size)
        return decrypted

