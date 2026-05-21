---
title: Lemons
description: Perhaps not everyone knows that Italy produces top-quality lemons. No wonder that forgetting them at the counter is a very serious matter...
date: 2023-12-10
tags:
  - web
  - robots.txt
order: 5
---

Perhaps not everyone knows that Italy produces top-quality lemons. No wonder that forgetting them at the counter is a very serious matter...  

[https://lemons.hackappatoi.com](https://lemons.hackappatoi.com)  

---

Check out the website. Nothing seems to stick out after checking Inspect. Maybe `robots.txt` has something?  

	User-agent: *
	Disallow: /flag
	
	User-agent: AdsBot-Google
	Disallow: /signora

Visiting `/flag` will redirect you to a Rick Roll, but visiting `/signora` will give you the flag!  

	hctf{e4sy_p3asy_th3_lem0ns_mada4a4aam!}
