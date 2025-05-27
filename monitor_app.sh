#!/bin/bash

# Monitor BTC Price Predictor application

echo "Monitoring BTC Price Predictor application..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a process is running
is_process_running() {
    pgrep -f "$1" >/dev/null
}

# Function to check if a port is in use
is_port_in_use() {
    netstat -tuln | grep ":$1 " >/dev/null
}

# Function to check API health
check_api_health() {
    curl -s "http://localhost:8000/health" | grep -q "status.*ok"
    return $?
}

# Function to check WebSocket health
check_websocket_health() {
    # This is a simple check that just verifies the port is open
    # A real check would involve connecting to the WebSocket
    is_port_in_use 8765
    return $?
}

# Function to check frontend health
check_frontend_health() {
    # This is a simple check that just verifies the port is open
    # A real check would involve making an HTTP request
    is_port_in_use 3000
    return $?
}

# Function to display status
display_status() {
    local service=$1
    local status=$2
    
    if [ "$status" -eq 0 ]; then
        echo "✅ $service: Running"
    else
        echo "❌ $service: Not running"
    fi
}

# Check if running in Docker
if command_exists docker && docker ps | grep -q "btc-predictor"; then
    echo "Application is running in Docker"
    
    # Check Docker containers
    echo "Docker containers:"
    docker ps --filter "name=btc-predictor"
    
    # Check Docker logs
    echo -e "\nRecent logs:"
    docker logs --tail 10 $(docker ps -q --filter "name=btc-predictor-backend")
    
    # Check API health
    if curl -s "http://localhost:8000/health" | grep -q "status.*ok"; then
        echo -e "\n✅ API: Healthy"
    else
        echo -e "\n❌ API: Unhealthy"
    fi
    
    # Check WebSocket
    if is_port_in_use 8765; then
        echo "✅ WebSocket: Running"
    else
        echo "❌ WebSocket: Not running"
    fi
    
    # Check Frontend
    if is_port_in_use 3000; then
        echo "✅ Frontend: Running"
    else
        echo "❌ Frontend: Not running"
    fi
else
    echo "Checking local processes..."
    
    # Check backend process
    is_process_running "python.*main.py"
    display_status "Backend" $?
    
    # Check frontend process
    is_process_running "node.*src/frontend"
    display_status "Frontend" $?
    
    # Check API health
    check_api_health
    display_status "API Health" $?
    
    # Check WebSocket health
    check_websocket_health
    display_status "WebSocket" $?
    
    # Check frontend health
    check_frontend_health
    display_status "Frontend Server" $?
    
    # Check log files
    echo -e "\nChecking log files..."
    if [ -f "logs/backend.log" ]; then
        echo "Recent backend logs:"
        tail -n 10 logs/backend.log
    else
        echo "❌ Backend log file not found"
    fi
    
    if [ -f "logs/frontend.log" ]; then
        echo -e "\nRecent frontend logs:"
        tail -n 10 logs/frontend.log
    else
        echo "❌ Frontend log file not found"
    fi
fi

# Check system resources
echo -e "\nSystem resources:"
echo "CPU usage:"
top -bn1 | head -n 5

echo -e "\nMemory usage:"
free -h

echo -e "\nDisk usage:"
df -h | grep -v "tmpfs"

echo -e "\nMonitoring completed!"
echo "For continuous monitoring, consider using tools like:"
echo "- Prometheus + Grafana"
echo "- ELK Stack (Elasticsearch, Logstash, Kibana)"
echo "- Datadog"
echo "- New Relic"