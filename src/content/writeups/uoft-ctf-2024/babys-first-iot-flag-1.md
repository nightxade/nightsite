---
title: Babys First IOT Flag 1
description: See introduction for complete context.
date: 2024-01-15
tags:
  - osint
  - iot
  - fcc-lookup
order: 6
---

See introduction for complete context.  

Part 1 - Here is an FCC ID, Q87-WRT54GV81, what is the frequency in MHz for Channel 6 for that device? Submit the answer to port 3895.  

---

Simple Google Search reveals that we can look up the FCC ID [here](https://fccid.io/). [This](https://fccid.io/Q87-WRT54GV81) is what we're looking for.  

Checking the RF Exposure Info and doing Ctrl+F for "channel", I quickly found that the frequency was 2437 MHz. Now, don't spend 20 minutes wondering why the submission wasn't working like I did, and check the discord to realize that you have to check the hint to get a valid submission.  

```
printf '2437\n\0' | nc 35.225.17.48 3895
```

And now we get the flag:  

    {FCC_ID_Recon}
