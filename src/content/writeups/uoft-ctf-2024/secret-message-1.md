---
title: Secret Message 1
description: We swiped a top-secret file from the vaults of a very secret organization, but all the juicy details are craftily concealed. Can you help me uncover them?
date: 2024-01-15
tags:
  - forensics
order: 5
---

We swiped a top-secret file from the vaults of a very secret organization, but all the juicy details are craftily concealed. Can you help me uncover them?  

Author: SteakEnthusiast  
[secret.pdf](/writeups/uoft-ctf-2024/assets/secret.pdf)  

---

Opening up the file, we see some redacted text in the PDF. Conveniently, though, we can highlight and copy text from the PDF. Simply highlighting the redacted area and doing Ctrl+C and pasting somewhere else gives us the flag!  

    uoftctf{fired_for_leaking_secrets_in_a_pdf}
