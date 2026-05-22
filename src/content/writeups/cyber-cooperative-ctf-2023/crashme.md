---
title: Crashme
description: Can you make this program crash?
date: 2023-12-19
tags:
  - pwn
order: 5
---

Can you make this program crash?  
`nc 0.cloud.chals.io 17289`  

[crashme](/writeups/cyber-cooperative-ctf-2023/assets/pwn/crashme)  
[crashme.c](/writeups/cyber-cooperative-ctf-2023/assets/pwn/crashme.c)  

---

We're given an ELF binary, a C source file, and a service to connect to. Here's `crashme.c`:  

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
    char buffer[32];
    printf("Give me some data: \n");
    fflush(stdout);
    fgets(buffer, 64, stdin);
    printf("You entered %s\n", buffer);
    fflush(stdout);
    return 0;
}
```

Seems like a very simple program. Simple programs require simple methods. `fgets()` can be vulnerable to buffer overflow, so why don't we try to send a large number of characters? I sent `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` to the local program, which immediately resulted in a `Segmentation fault`.  

This is what we need then! Send the same input to the program to get the flag:  

    flag{segfaults_a_hackers_best_friend}
