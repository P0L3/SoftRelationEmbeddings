export PYTHONPATH=`pwd`/src/main/python:${PYTHONPATH}
python3 -m experiment "${@}"