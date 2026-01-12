#!/bin/bash
# Startup script for Azure App Service

# Install system dependencies if needed
# Note: Azure App Service already has most required libraries

# Start Gunicorn server
gunicorn --bind=0.0.0.0:8000 --timeout=600 --workers=4 web_app:app
