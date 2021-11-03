FROM python:3.9-slim-buster

# Create app directory
WORKDIR /usr/src/app/

# Install app dependencies
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN export FLASK_APP=autoARIMA

# Run
EXPOSE 5002
COPY ../services/autoARIMA.py ./
COPY ../services/utility.py ./
CMD [ "python", "autoARIMA.py"]