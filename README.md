# Matrix Operations API

Matrix Operations API is a Django-based application for performing various matrix operations such as addition, subtraction, multiplication, and transposition.

---

## Requirements

- Python 3.7+
- Django 3.2+
- pip (Python package manager)

---

## Setup Instructions

### 1. Clone the Repository

Begin by cloning the repository to your local machine and navigating to the project directory.

git clone https://github.com/mawedi/matrix-operations-api.git
cd matrix-operations-api

### 2. Create a Virtual Environment and activate it

python -m venv env
.\env\Scripts\activate

### 3. Installing requriements

pip install -r requirements.txt

### 4. Migrating data to the database

python manage.py makemigrations
python manage.py migrate

### 5. Running server

python manage.py runserver 0.0.0.0:8000

### 6. Interaction with api

http://127.0.0.1:8000/
http://your_ip_address:8000/

### 7. Endpoints

you can find Endpoints to interact with the api in /operations/urls
