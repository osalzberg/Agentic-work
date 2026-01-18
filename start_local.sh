#!/bin/bash
# Start local web app with current user's Azure credentials

cd "$(dirname "$0")"

# Kill any existing process on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

# Check if Azure CLI is logged in
if ! az account show &>/dev/null; then
    echo "❌ Not logged into Azure CLI. Please run: az login"
    exit 1
fi

echo "✅ Azure credentials found"
echo "📊 Starting web app with your Azure credentials..."

# Activate venv and start app
source venv/bin/activate
python web_app.py
