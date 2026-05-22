---
title: Crypto Yors Truly
description: I have encrypted some text but it seems I have lost the key! Can you find it?
date: 2024-03-19
tags:
  - crypto
order: 2
---

I have encrypted some text but it seems I have lost the key! Can you find it?  

[yors-truly.py](/writeups/wolv-ctf-2024/assets/beginner/yors-truly.py)  

---
We're provided a Python source file:  

```py
import base64

plaintext = "A string of text can be encrypted by applying the bitwise XOR operator to every character using a given key"
key = "" # I have lost the key!

def byte_xor(ba1, ba2):
    return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])

ciphertext_b64 = base64.b64encode(byte_xor(key.encode(), plaintext.encode()))

ciphertext_decoded = base64.b64decode("NkMHEgkxXjV/BlN/ElUKMVZQEzFtGzpsVTgGDw==")


print(ciphertext_decoded)
```

Note that XOR is a reversible operation, since `a ^ a = 0` and XOR is associative and commutative.  

Therefore, to recover the key, it we can simply XOR the plaintet and the ciphertext. See below:  

$$plaintext=pt$$  
$$ciphertext=ct$$  
$$ct=pt \oplus key$$  
$$ct \oplus pt = pt \oplus key \oplus pt = key$$  

Therefore, here is the solve script:  

```py
import base64

plaintext = "A string of text can be encrypted by applying the bitwise XOR operator to every character using a given key"
key = "" # I have lost the key!

def byte_xor(ba1, ba2):
    return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])

ciphertext_b64 = base64.b64encode(byte_xor(key.encode(), plaintext.encode()))

ciphertext_decoded = base64.b64decode("NkMHEgkxXjV/BlN/ElUKMVZQEzFtGzpsVTgGDw==")

print(byte_xor(plaintext.encode(), ciphertext_decoded))
```

Run the script to get the flag!  

    wctf{X0R_i5_f0rEv3r_My_L0Ve}
