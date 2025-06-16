# Use an official Python runtime as a parent image
# We are using a lightweight Python 3.9 image.
# Using a specific version tag (like 3.9-slim) is recommended for stability
# and to ensure your build is reproducible.
FROM python:3.9-slim

# Set environment variables
# These environment variables are commonly used for Python applications
# running in Docker to ensure output is unbuffered and to manage pip behavior.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set the working directory in the container
# This sets the current directory inside the container to /app.
# All subsequent commands like COPY, RUN, and CMD will be executed relative to this directory.
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
# This is an optimization. If only your code changes, Docker can use the cached
# layer for dependency installation, speeding up subsequent builds.
COPY requirements.txt .

# Install the Python dependencies
# This command executes during the image build. It reads the requirements.txt file
# and installs all the listed Python packages using pip.
# Make sure you have 'Flask', 'numpy', 'torch', 'gunicorn', and any other
# necessary libraries listed in your requirements.txt file.
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
# This copies all other files and directories from your local project's root
# (where the Dockerfile is located) into the /app directory inside the container.
# This includes your main Flask file (your_app.py), the utils directory,
# the model files (utils/model/model.pth, utils/model/model.py), etc.
COPY . /app

# Expose the port your Flask app will run on
# This instruction informs Docker that the container listens on port 7860.
# Hugging Face Spaces Docker SDK typically expects applications to listen on port 7860.
# This doesn't actually publish the port, but serves as documentation.
EXPOSE 7860

# Command to run your application when the container launches
# This is the default command that will be executed when a container is started
# from this image. We use gunicorn, a popular WSGI server for Python web apps.
# It tells gunicorn to run the 'app' object found in your 'your_app.py' file.
# '--workers 4': Specifies the number of worker processes for gunicorn. Adjust as needed.
# '--bind 0.0.0.0:7860': Binds gunicorn to all network interfaces on port 7860.
# 'your_app:app': The format is [module_name]:[variable_name].
#   - 'your_app' should be the name of your main Python file (without the .py extension).
#   - 'app' should be the name of your Flask application instance
#     (e.g., app = flask.Flask(__name__)).
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:7860", "flask_Character:app"]

# Important Note:
# Before building your Docker image, make sure to remove the following block
# from your main Flask application file (your_app.py):
#
# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000)
#
# This block is for running the Flask development server directly,
# but when using a WSGI server like Gunicorn in production (or in Docker),
# Gunicorn handles starting the application.
