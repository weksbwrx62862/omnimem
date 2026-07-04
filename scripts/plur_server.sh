#!/bin/bash
# Plur 服务器启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
HOST=${PLUR_HOST:-"0.0.0.0"}
PORT=${PLUR_PORT:-8080}
LOG_LEVEL=${PLUR_LOG_LEVEL:-"INFO"}
PID_FILE="/tmp/plur_server.pid"
LOG_FILE="/tmp/plur_server.log"

# 函数：打印消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函数：检查端口是否被占用
check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_error "Port $PORT is already in use"
        return 1
    fi
    return 0
}

# 函数：启动服务器
start_server() {
    print_info "Starting Plur server..."
    
    # 检查端口
    if ! check_port; then
        exit 1
    fi
    
    # 获取脚本目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # 启动服务器
    cd "$SCRIPT_DIR"
    nohup python3 plur_server.py --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL" > "$LOG_FILE" 2>&1 &
    
    # 保存 PID
    echo $! > "$PID_FILE"
    
    # 等待启动
    sleep 2
    
    # 检查是否启动成功
    if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        print_info "Plur server started successfully"
        print_info "PID: $(cat $PID_FILE)"
        print_info "URL: http://$HOST:$PORT"
        print_info "Log: $LOG_FILE"
    else
        print_error "Failed to start Plur server"
        cat "$LOG_FILE"
        exit 1
    fi
}

# 函数：停止服务器
stop_server() {
    print_info "Stopping Plur server..."
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm -f "$PID_FILE"
            print_info "Plur server stopped"
        else
            print_warn "Plur server is not running"
            rm -f "$PID_FILE"
        fi
    else
        print_warn "PID file not found"
    fi
}

# 函数：重启服务器
restart_server() {
    stop_server
    sleep 1
    start_server
}

# 函数：查看状态
status_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            print_info "Plur server is running"
            print_info "PID: $PID"
            print_info "URL: http://$HOST:$PORT"
            
            # 获取统计信息
            echo ""
            print_info "Server stats:"
            curl -s "http://localhost:$PORT/stats" | python3 -m json.tool 2>/dev/null || print_warn "Failed to get stats"
        else
            print_warn "Plur server is not running (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        print_warn "Plur server is not running"
    fi
}

# 函数：查看日志
logs_server() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        print_warn "Log file not found"
    fi
}

# 函数：健康检查
health_check() {
    print_info "Performing health check..."
    
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        print_info "Health check passed"
        curl -s "http://localhost:$PORT/health" | python3 -m json.tool
    else
        print_error "Health check failed"
        exit 1
    fi
}

# 函数：显示帮助
show_help() {
    echo "Usage: $0 {start|stop|restart|status|logs|health|help}"
    echo ""
    echo "Commands:"
    echo "  start   - Start the Plur server"
    echo "  stop    - Stop the Plur server"
    echo "  restart - Restart the Plur server"
    echo "  status  - Show server status"
    echo "  logs    - Show server logs"
    echo "  health  - Perform health check"
    echo "  help    - Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  PLUR_HOST      - Host to bind (default: 0.0.0.0)"
    echo "  PLUR_PORT      - Port to bind (default: 8080)"
    echo "  PLUR_LOG_LEVEL - Log level (default: INFO)"
}

# 主逻辑
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        status_server
        ;;
    logs)
        logs_server
        ;;
    health)
        health_check
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
