[Deep RL Project Calendar](https://calendar.google.com/calendar/ical/fuc8n5c750pte22c32kgi940ug%40group.calendar.google.com/private-42a8e26f0d1130a135a2a1fd08accb6a/basic.ics)


## OpenAI gym installation
IF pachi-py fails to build wheel:
use `MACOSX_DEPLOYMENT_TARGET=10.12 pip install -e .[all]` (for bash/zsh)
or `env MACOSX_DEPLOYMENT_TARGET=10.12 pip install -e .[all] (for fish)`
... installer seems to assume OSX 10.5 as build env by default.
