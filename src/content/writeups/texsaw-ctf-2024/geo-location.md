---
title: Geo Location
description: Find what street this picture was taken from.
date: 2024-03-25
tags:
  - osint
order: 3
---

Find what street this picture was taken from.  

Format the flag as the following: The street name in all caps with the spaces replaced by underscores.  

Example: If the street was Bourbon Street the flag would be: texsaw{BOURBON_STREET}  

[picture.jpg](/writeups/texsaw-ctf-2024/assets/picture.jpg)  

---

Do a Google Reverse Image Search, drawing a rectangle area including only the most prominent building. Going to "Exact Matches", you'll find a building called TCC Legacy Kincaid. Searching it up will provide us an address. Move around the address in Google Street View until you find the Beal Bank sign, and thus the road it was taken from -- Legacy Dr!  

    texsaw{LEGACY_DRIVE}
