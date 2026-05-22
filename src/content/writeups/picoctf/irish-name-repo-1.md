---
title: Irish Name Repo 1
description: "There is a website running at https://jupiter.challenges.picoctf.org/problem/50009/ (link) or http://jupiter.challenges.picoctf.org:50009. Do you think you can log us in? Try to see if you can login!"
date: 2023-07-01
tags:
  - web
order: 28
---

There is a website running at https://jupiter.challenges.picoctf.org/problem/50009/ ([link](https://jupiter.challenges.picoctf.org/problem/50009/)) or http://jupiter.challenges.picoctf.org:50009. Do you think you can log us in? Try to see if you can login!  

---

Navigating to the menu and checking out the Support section, notice that the first answer states “...I keep getting something called a SQL Error.” So we need to perform an SQL injection!  

Navigate to the Login page. Instead of trying to find out what the admin username and password is, let’s make it so that the SQL check will always evaluate as true! Put anything (or nothing) in the username text box, but for the password, input `'or'1'='1`.  

Short explanation for input:  

The SQL username and password check will look something like this, most likely:  
```sql
SELECT * FROM admin_table WHERE username = '[username]' AND password = '[password]'
```
Using our input, it will look like this:  
```sql
SELECT * FROM admin_table WHERE username = '' AND password = ''or'1'='1'
```
The first character of our input, the single quote, closes the opening quote surrounding our input. Then, we include an or statement that checks if `'1'='1'`, where the last single quote is from the closing quote originally surrounding our input, that always evaluates as true, meaning this statement will select every entry in the table regardless if the username and password are correct or not.  

After logging in with the payload, you should receive the flag!  

    picoCTF{s0m3_SQL_fb3fe2ad}
