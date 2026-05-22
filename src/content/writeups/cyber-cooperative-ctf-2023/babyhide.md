---
title: Babyhide
description: This little baby is figuring out how to computer! It looks like the baby hid some of my files though. I have no idea what to do, can you get my files back?
date: 2023-12-19
tags:
  - forensics
order: 2
---

This little baby is figuring out how to computer! It looks like the baby hid some of my files though. I have no idea what to do, can you get my files back?

[babyhide.jpg](/writeups/cyber-cooperative-ctf-2023/assets/forensics/babyhide.jpeg)  

---

Running `binwalk` on the file tells us there is a PDF file located at offset 0x1CAB6. We can write a Python program to extract it:  

```py
r = open('babyhide.jpeg', 'rb').read()
w = open('babyhide.pdf', 'wb')

w.write(r[0x1CAB6:])
w.close()
```

Opening up the pdf gets us the flag!  

    flag{baby_come_back}
