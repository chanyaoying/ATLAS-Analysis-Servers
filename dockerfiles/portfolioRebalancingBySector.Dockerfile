FROM python:3.9-slim-buster

# Create app directory
WORKDIR /usr/src/app/

# Install app dependencies
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN export FLASK_APP=portfolioRebalancingBySector

# Run
EXPOSE 5006
COPY ./services/portfolioRebalancingBySector.py ./
COPY ./services/MPT_functions.py ./
COPY ./services/utility.py ./

# CMD [ "python", "portfolioRebalancingBySector.py"]
CMD [ "gunicorn", "-b", "0.0.0.0:5006", "portfolioRebalancingBySector:app", "-w", "4"]