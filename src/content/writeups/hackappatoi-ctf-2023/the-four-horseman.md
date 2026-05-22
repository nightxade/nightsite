---
title: The Four Horseman
description: The Cavaliere is returned. He is awake and he’s ready to unleash the apocalypse. Are you the chosen one? Solving this challenge will give you the access to the war against the four horsemen. Be ready.
date: 2023-12-10
tags:
  - rev
order: 4
---

The Cavaliere is returned. He is awake and he’s ready to unleash the apocalypse. Are you the chosen one? Solving this challenge will give you the access to the war against the four horsemen. Be ready.  

[thefourhorsemen](/writeups/hackappatoi-ctf-2023/assets/thefourhorsemen)

---

Let's head over to [Dogbolt](https://dogbolt.org/?id=f716f53f-44ad-42f7-8f53-60b52a5358ff#Hex-Rays=158) to decompile this.  

Immediately, in the Hex-Rays decompilation, I found this string:  

    upgs{lbher_ernql_gb_fgbc_gur_ncbpnylcfr}

Hm. Seems to be encrypted. Maybe it's just some ROT offset?  

Heading over to [https://www.dcode.fr/rot-cipher](https://www.dcode.fr/rot-cipher) and using ROT13 gives us the flag!  

	hctf{youre_ready_to_stop_the_apocalypse}
