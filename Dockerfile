## Use an official Python runtime as a base image
FROM python:2.7

COPY requirements.txt /app/requirements.txt

# Set the working directory to /app
WORKDIR /app

RUN pip install -U pip setuptools

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
ADD . /app

# Make port 22 available to the world outside this container
EXPOSE 22

# Run app.py when the container launches
CMD ["python", "hog-iterator.py"]