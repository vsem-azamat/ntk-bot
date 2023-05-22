# NTK bot
Telegram bot in @chat_ntk chat for students who regularly visit the National Technical Library.

## Current and planned functions:
- [x] Shows the current number of people in the NTK
- [x] Regular storage of data from the library website on the number of people
- [x] Draws a diagram of people's visits in the NTK
- [x] Anonymous questions in [@chat_ntk](t.me/chat_ntk)
- [-] Predicting the number of people in the library based on the received data
- [-] Daily reports for the morning weather forecast for the whole day
- [-] Anti-bot filter
- [-] Function for temporary self muta/ban from the chat so that students are not distracted from their studies

## Installation and start

### Necessary:
Install requirements packages
```sh
> python3 -m venv venv
> pip install -r requirements.txt
```
Create `.env` file and write **api token**
```env
BOT_TOKEN=<TOKEN>
```

### Optional:
Additional adjustable values in `.env`
```env
DELTA_TIME=<int>
SUPER_ADMINS=<int,int,int,...>
```
* `DELTA_TIME` - The time interval with which the bot collects visit data from the site. The default value is `20`
* `SUPER_ADMINS` - List of super admins for admin commands 

#### Anonymous emails in chat:
To set up a bad word filter for **anonymous messages**, create a `bad_words.txt` file and write the necessary words on each new line. When a bad word is detected, the bot does not send an anonymous letter to the chat, but sends it to the bot administrator with the sender's data.

```txt
aboba
lupa
pupa
```

## Commands:
Prefixes: `!/`
- `/ntk` - Show the current number of people in the library
- `/anon <text>` - To write a private message to the bot to send an anonymous message to the chat @chat_ntk