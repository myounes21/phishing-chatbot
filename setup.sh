#!/bin/bash

# Phishing Campaign Analyzer - Setup Script
# This script helps you set up the environment and dependencies

set -e

echo "=========================================="
echo "Phishing Campaign Analyzer - Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.9 or higher is required${NC}"
    echo "Current version: $python_version"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ pip3 is installed${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data/raw data/processed data/insights logs
echo -e "${GREEN}✓ Directories created${NC}"

# Check for environment file
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Qdrant Configuration (optional, will use in-memory if not available)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Optional: Qdrant API Key
# QDRANT_API_KEY=your_qdrant_api_key
EOF
    echo -e "${YELLOW}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠  Please edit .env and add your Groq API key${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Check for Docker
echo ""
echo "Checking for Docker..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker is installed${NC}"
    
    if command -v docker-compose &> /dev/null; then
        echo -e "${GREEN}✓ Docker Compose is installed${NC}"
    else
        echo -e "${YELLOW}⚠  Docker Compose is not installed${NC}"
    fi
else
    echo -e "${YELLOW}⚠  Docker is not installed${NC}"
    echo "   You can still use the application without Docker"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Add your Groq API key to the .env file:"
echo "   ${YELLOW}nano .env${NC}"
echo ""
echo "2. (Optional) Start Qdrant with Docker:"
echo "   ${YELLOW}docker run -p 6333:6333 qdrant/qdrant${NC}"
echo "   Or the system will use in-memory storage automatically"
echo ""
echo "3. Run the Streamlit app:"
echo "   ${YELLOW}source venv/bin/activate${NC}"
echo "   ${YELLOW}streamlit run chatbot_app.py${NC}"
echo ""
echo "   OR run the CLI:"
echo "   ${YELLOW}python chatbot_cli.py --csv sample_phishing_data.csv${NC}"
echo ""
echo "4. (Optional) Use Docker Compose for everything:"
echo "   ${YELLOW}docker-compose up -d${NC}"
echo ""
echo "For more information, see README.md"
echo ""
