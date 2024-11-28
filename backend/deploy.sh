#!/bin/bash
gunicorn --bind 0.0.0.0:5000 app:app --workers 4 --timeout 120