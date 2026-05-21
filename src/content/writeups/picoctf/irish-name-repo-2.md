---
title: Irish Name Repo 2
description: "There is a website running at https://jupiter.challenges.picoctf.org/problem/53751/ (link). Someone has bypassed the login before, and now it's being strengthened. Try to see if you can still login! or http://jupiter.challenges.picoctf.org:53751"
date: 2023-07-01
tags:
  - web
  - sql-injection
order: 29
---

There is a website running at https://jupiter.challenges.picoctf.org/problem/53751/ ([link](https://jupiter.challenges.picoctf.org/problem/53751/)). Someone has bypassed the login before, and now it's being strengthened. Try to see if you can still login! or http://jupiter.challenges.picoctf.org:53751  

---

Using the same input as in Irish-Name-Repo 1 returns the following: “SQLi detected.” So it seems like we can’t include any SQL in our parameters.  
Instead, however, maybe we can exploit SQL comments!  

The SQL for this challenge is likely similar, i.e.  
```sql
SELECT * FROM admin_table WHERE username = '[username]' AND password = '[password]'
```
*(Note that this can actually be confirmed by setting the debug parameter to 1 using a tool such as Burp Suite or simply Terminal)*

Hence, in the username input, we can input `admin'--`, which will change the SQL to this:  
```sql
SELECT * FROM admin_table WHERE username = 'admin''-- AND password = '' 
```
The `--` is an SQL comment, and comments out the rest of the SQL such that it never checks if the password is correct. Enter the input in, and get your flag!  

    picoCTF{m0R3_SQL_plz_c34df170}
