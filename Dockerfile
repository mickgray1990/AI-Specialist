# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the local directory content to the container
COPY . .

# Install the necessary packages
RUN pip install streamlit openai pandas numpy PyPDF2

# Specify the port your Streamlit app will listen on
EXPOSE 8501

# Command to run the application when the container starts
CMD ["streamlit", "run", "ai_pdf_chatbot.py"]
