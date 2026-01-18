#!/usr/bin/env python3
"""
Startup script for Azure App Service
"""
import sys
import os

# Ensure we're in the right directory
os.chdir('/home/site/wwwroot')
sys.path.insert(0, '/home/site/wwwroot')

# Import and run the app
from web_app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
