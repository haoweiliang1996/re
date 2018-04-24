#export FLASK_APP=root.py 
#flask run --host=0.0.0.0
gunicorn -w 1 root:app #--threads 2
