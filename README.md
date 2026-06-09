# NTK bot
Telegram bot in @chat_ntk chat for students who regularly visit the National Technical Library. The bot regularly collects statistics on library visits. Based on this data, simple machine learning models were created for regression of visits.


| ![Daily graph with predictions](example_images/daily_graph_with_predictions.jpg) | ![Weather forecast](example_images/weather_forecast.jpg) |
|:---:|:---:|
| Daily graph with predictions | Weather forecast |


## Current and planned functions:
- [x] Shows the current number of people in the NTK
- [x] Regular storage of data from the library website on the number of people
- [x] Draws a diagram of people's visits in the NTK
- [x] Predicting the number of people in the library based on the received data with ML models
- [x] Weather forecasts
- [ ] Anti-bot filter
- [ ] Function for temporary self mute/ban from the chat so that students are not distracted from their studies

## Data sources:
- [NTK website](https://www.techlib.cz/)
- [Open-Meteo](https://open-meteo.com/)


## How prediction works:
The data of visits to the National Technical Library is permanently stored. This data was processed and fed to models for training according to this principle:

| X1 | X2 | X3 | X4 | Y |
|:---:|:---:|:---:|:---:| :---:|
| day of the year | day of the week | time | month | number of people |

`f(X1, X2, X3, X4) = Y -> f(day of the year, day of the week, time, month) = number of people`

Two models are used:
- Random Forest Regressor
- Gradient Boosting Regressor

## Installation and start

The project is managed with [uv](https://docs.astral.sh/uv/).

### Necessary:
Install dependencies (creates a virtual environment from `uv.lock`)
```sh
> uv sync
```
Create a `.env` file and add the **bot token**
```env
BOT_TOKEN=<TOKEN>
```

### Start:
From the root directory of the project
```sh
> uv run ntk-bot
```
(equivalently `uv run python -m bot`)

### Optional:
Additional adjustable values in `.env`
```env
DELTA_TIME=<int>
SUPER_ADMINS=<int,int,int,...>
OPENROUTER_API_KEY=<KEY>
OPENROUTER_MODEL=<slug>
ANSWER_PROBABILITY=<float>
```
* `DELTA_TIME` - The time interval with which the bot collects visit data from the site. The default value is `20`
* `SUPER_ADMINS` - List of super admins for admin commands
* `OPENROUTER_API_KEY` - [OpenRouter](https://openrouter.ai/) key used for the random GPT replies
* `OPENROUTER_MODEL` - Model slug passed to OpenRouter. Default `openai/gpt-4o`
* `ANSWER_PROBABILITY` - Probability of a random GPT reply. Default `0.025`

### Development:
```sh
> uv run ruff format .      # format
> uv run ruff check .       # lint
> uv run ty check           # type-check
> uv run pytest             # tests
```


## Commands:
Prefixes: `!/`
- `/ntk` - Show the current number of people in the library
- `/help` - Show help
- `/graph` - Draw and send a diagram of library visits
- `/learn` - Train (re-) regression ML models for predicting the number of people in the library
- `/weather` - Show weather forecast