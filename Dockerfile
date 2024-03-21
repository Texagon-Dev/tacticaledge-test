FROM ashokpant/python-3.10-cv

ARG PORT
ARG OPENAI_API_KEY
ARG SUPABASE_KEY
ARG SUPABASE_URL

# Expose the specified port
EXPOSE $PORT

# Set the working directory inside the container
WORKDIR /backend

# Copy the requirements.txt file into the container
COPY requirements.txt .


# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

#RUN pip uninstall --yes opencv-python
#RUN pip uninstall --yes opencv-python-headless
#RUN pip install opencv-python-headless


# Copy the rest of the application code into the container
COPY . .

# Specify the default command to run when the container starts
EXPOSE $PORT
CMD ["sh", "-c", "streamlit --server.port $PORT --server.address 0.0.0.0 --server.enableCORS true interface.py"]
