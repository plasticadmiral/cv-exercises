# Frequently Asked Questions.


## Setup: 

### - Regarding bashprofile and bashrc:
- Problem: conda: Befehl nicht gefunden / Command not found
- Actual problem: .bashrc is not sourced when logging in via ssh
- see [bashrc-at-ssh-login](https://stackoverflow.com/questions/820517/bashrc-at-ssh-login)
- Solution:
- if .bash_profile does not exist next to your bash file then create it and paste the following
```
if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi
```
