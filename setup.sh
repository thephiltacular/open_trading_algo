#!/bin/bash

# Open Trading Algo Setup Script
# This script provides an alternative to Makefile commands for setting up the repository

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
}

# Setup Poetry
setup_poetry() {
    log_info "Setting up Poetry..."
    if ! command_exists poetry; then
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        log_success "Poetry installed"
    else
        log_info "Poetry already installed"
    fi
    poetry config virtualenvs.in-project true
}

# Setup configuration files
setup_config() {
    log_info "Setting up configuration files..."
    if [ ! -f "secrets.env" ]; then
        cp secrets.env.example secrets.env
        log_success "Created secrets.env from template"
        log_warning "Please edit secrets.env and add your API keys"
    else
        log_info "secrets.env already exists"
    fi

    if [ ! -f ".env" ]; then
        cp .env.example .env 2>/dev/null || echo "# Add your environment variables here" > .env
        log_success "Created .env file"
    else
        log_info ".env already exists"
    fi

    # Create data directories
    mkdir -p data/cache/sqlite
    mkdir -p data/cache/parquet
    mkdir -p data/cache/influxdb
    mkdir -p logs
    log_success "Data directories created"
}

# Setup SQLite database
setup_sqlite() {
    log_info "Setting up SQLite database..."
    if [ ! -f "data/tv_data_cache.sqlite3" ]; then
        touch data/tv_data_cache.sqlite3
        log_success "SQLite database created"
    else
        log_info "SQLite database already exists"
    fi
}

# Setup InfluxDB
setup_influxdb() {
    log_info "Setting up InfluxDB..."
    if command_exists docker; then
        if [ "$(docker ps -q -f name=influxdb)" ]; then
            log_info "InfluxDB container already running"
        elif [ "$(docker ps -aq -f status=exited -f name=influxdb)" ]; then
            docker start influxdb
            log_success "InfluxDB container started"
        else
            docker run -d --name influxdb -p 8086:8086 influxdb:2.0
            log_success "InfluxDB container created and started"
            log_info "Waiting for InfluxDB to be ready..."
            sleep 5
        fi

        # Initialize InfluxDB
        if curl -X POST http://localhost:8086/api/v2/setup \
            -H "Content-Type: application/json" \
            -d '{"username": "admin", "password": "password", "org": "trading", "bucket": "market_data", "token": "my-token"}' 2>/dev/null; then
            log_success "InfluxDB initialized"
        else
            log_warning "Could not initialize InfluxDB (may already be initialized)"
        fi
    else
        log_warning "Docker not found. Please install Docker to use InfluxDB."
        log_info "Manual setup: docker run -d --name influxdb -p 8086:8086 influxdb:2.0"
    fi
}

# Install dependencies
install_deps() {
    log_info "Installing dependencies..."
    poetry install
    log_success "Dependencies installed"

    log_info "Setting up pre-commit hooks..."
    poetry run pre-commit install
    poetry run pre-commit run --all-files
    log_success "Pre-commit hooks configured"
}

# Main setup function
main() {
    echo -e "${BLUE}🚀 Open Trading Algo Setup${NC}"
    echo "=========================="

    case "${1:-all}" in
        "venv")
            setup_venv
            ;;
        "poetry")
            setup_poetry
            ;;
        "config")
            setup_config
            ;;
        "sqlite")
            setup_sqlite
            ;;
        "influxdb")
            setup_influxdb
            ;;
        "deps")
            install_deps
            ;;
        "all")
            setup_venv
            setup_poetry
            setup_config
            setup_sqlite
            setup_influxdb
            install_deps
            log_success "Complete setup finished!"
            echo ""
            log_info "Next steps:"
            echo "  1. Edit secrets.env and add your API keys"
            echo "  2. Run: source .venv/bin/activate"
            echo "  3. Run: make dev_env"
            ;;
        *)
            log_error "Usage: $0 {venv|poetry|config|sqlite|influxdb|deps|all}"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
