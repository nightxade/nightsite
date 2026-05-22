---
title: Inside Bear
description: "This is the hardest reversing challenge I've ever met!!! Can you please help me? It even has source code attached to it."
date: 2023-12-11
tags:
  - misc
order: 5
---

This is the hardest reversing challenge I've ever met!!! Can you please help me? It even has source code attached to it.  

[31201020812a2cc96988054c9661143d.zip](/writeups/ping-ctf-2023/assets/31201020812a2cc96988054c9661143d.zip)  

---

We're given two files, `look-inside` and `look-inside.c`. Here's `look-inside.c`:  

```c
#include <stdio.h>

int main() {
   printf("Hi mom!\n");
   return 0;
}
```

Hm. Seems like nothing to see here. Well, the challenge hints towards looking inside, so maybe we can find something if we run `strings`?  

I ran `strings look-inside | less` and immediately found something suspicious:  

`inside/CAPTURED_TRANSMISSION.wav`  

So there are files stored within this file. With either `binwalk -eM` (`-e` for extract and `-M` for recursive) or 7-Zip File Manager, you can extract these.  

The extracted files are:  

    CAPTURED_TRANSMISSION.wav
    static.ogg
    you-died.gif

grepping for `ping` returned nothing, so I took a look at all the files. I actually immediately noticed that `CAPTURED_TRANSMISSION.wav` sounded very much like another challenge I had done only a week ago in one of Newport Blake CTF 2023's misc challenges. Therefore, I immediately knew to open Audacity and take a look at the spectrogram. This is a very common tactic to pass messages through audio files in CTFs, so it is definitely recommended to keep it in mind on forensics challenges.  

The spectrogram contained the following message:  

`cGluZ3tJX2Fsd2F5JF9jMG1lX2JAY2t9`

This is clearly base64 encoding, and decoding it produces the flag!  

    ping{I_alway$_c0me_b@ck}
