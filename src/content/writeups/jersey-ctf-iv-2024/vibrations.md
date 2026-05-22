---
title: Vibrations
description: While managing some of our thermal systems, we noticed unusual behavior. We collected the network traffic, see if you can find anything unusual.
date: 2024-03-25
tags:
  - forensics
  - network
order: 5
---

While managing some of our thermal systems, we noticed unusual behavior. We collected the network traffic, see if you can find anything unusual.  

Developed by: Dan D  

[final.pcapng](/writeups/jersey-ctf-iv-2024/assets/final.pcapng)  

---

Follow TCP Stream --> Stream 1 --> Hex Dump  

Some bytes left out --> all the bytes after the "}" are filled into the missing byte places in sequential order.  

    jctf{I_rEllAy_H0p3_thi$_i$nt_a_p0ol_sy$t3m_aGa1n}
