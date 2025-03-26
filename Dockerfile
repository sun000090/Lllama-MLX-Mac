FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get -y install libgl1-mesa-dev

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN useradd -m -u 58241 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN mkdir ~/.cache

RUN chmod -R 777 ~/.cache

COPY . .

CMD [ "gunicorn", "-b", "0.0.0.0:7860", "--timeout", "300", "main:app"]