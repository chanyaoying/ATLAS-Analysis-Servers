FROM python:3.9-slim-buster

# Create app directory
WORKDIR /usr/src/app/

# Install app dependencies
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN export FLASK_APP=companyInfo

# Run
EXPOSE 5001
COPY ../services/companyInfo.py ./
CMD [ "python", "companyInfo.py"]