#!/bin/bash
# 共享记忆服务启动脚本

set -e

# 配置
PLUR_PORT=8080
PLUR_HOST="0.0.0.0"
PID_FILE="/tmp/plur_server.pid"
LOG_FILE="/tmp/plur_server.log"

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# 启动服务
start() {
    print_info "启动共享记忆服务..."

    # 检查是否已运行
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        print_warn "服务已在运行 (PID: $(cat $PID_FILE))"
        return 1
    fi

    # 启动 Plur 服务器
    cd ~/.hermes/plugins/omnimem
    nohup python3 core/plur_server.py --host "$PLUR_HOST" --port "$PLUR_PORT" --log-level INFO > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    sleep 2

    # 检查启动状态
    if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        print_info "✅ Plur 服务器已启动"
        print_info "   PID: $(cat $PID_FILE)"
        print_info "   地址: http://$PLUR_HOST:$PLUR_PORT"
        print_info "   日志: $LOG_FILE"
    else
        print_error "❌ 启动失败"
        cat "$LOG_FILE"
        return 1
    fi

    # 测试连接
    if curl -s "http://localhost:$PLUR_PORT/health" > /dev/null 2>&1; then
        print_info "✅ 健康检查通过"
    else
        print_error "❌ 健康检查失败"
        return 1
    fi
}

# 停止服务
stop() {
    print_info "停止共享记忆服务..."

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm -f "$PID_FILE"
            print_info "✅ 服务已停止"
        else
            print_warn "服务未运行"
            rm -f "$PID_FILE"
        fi
    else
        print_warn "PID 文件不存在"
    fi
}

# 重启服务
restart() {
    stop
    sleep 1
    start
}

# 查看状态
status() {
    print_info "共享记忆服务状态:"

    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        PID=$(cat "$PID_FILE")
        print_info "✅ 运行中 (PID: $PID)"
        print_info "   地址: http://localhost:$PLUR_PORT"

        # 获取统计信息
        echo ""
        print_info "服务器统计:"
        curl -s "http://localhost:$PLUR_PORT/stats" | python3 -m json.tool 2>/dev/null || print_warn "无法获取统计信息"
    else
        print_warn "❌ 未运行"
    fi
}

# 查看日志
logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        print_warn "日志文件不存在"
    fi
}

# 健康检查
health() {
    print_info "健康检查..."

    if curl -s "http://localhost:$PLUR_PORT/health" | python3 -m json.tool; then
        print_info "✅ 健康检查通过"
    else
        print_error "❌ 健康检查失败"
        return 1
    fi
}

# 帮助
help() {
    echo "用法: $0 {start|stop|restart|status|logs|health|help}"
    echo ""
    echo "命令:"
    echo "  start   - 启动共享记忆服务"
    echo "  stop    - 停止共享记忆服务"
    echo "  restart - 重启共享记忆服务"
    echo "  status  - 查看服务状态"
    echo "  logs    - 查看日志"
    echo "  health  - 健康检查"
    echo "  help    - 显示帮助"
}

# 主逻辑
case "$1" in
    start) start ;;
    stop) stop ;;
    restart) restart ;;
    status) status ;;
    logs) logs ;;
    health) health ;;
    help|--help|-h) help ;;
    *) print_error "未知命令: $1"; help; exit 1 ;;
esac
