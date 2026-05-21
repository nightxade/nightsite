---
title: Babbling About
description: They say a monkey with a typewriter will eventually type out a masterpiece. But what about a flag?
date: 2023-12-04
tags:
  - osint
  - babel
order: 8
---

They say a monkey with a typewriter will eventually type out a masterpiece. But what about a flag?  

Enclose the flag in brackets.  

I think it was on the same page that websites return when the page doesnt exist?  

[location.txt](/writeups/newport-blake-ctf-2023/assets/osint/location.txt)  

---

`location.txt` seemed to imply that this was some sort of book, so I Googled about Hexagon code books. I ended up finding the *Library of Babel*, which would make sense given the name and description of the challenge.  

Putting in the hexagon code and evaluating the wall, shelf, and volume, I found the book. I downloaded it and Ctrl+F'd for `nbctf` and found the following: `nbctf love.this.library`.  

P.S. the book is [here](https://libraryofbabel.info/referencehex.html)  

Thus, the flag is:  

    nbctf{love.this.library}
