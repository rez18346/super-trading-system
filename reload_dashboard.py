#!/usr/bin/env python3
"""
🔄 Reload control_api модуля в рантайме (без перезапуска трейдера)
Перезагружает модуль и перезапускает uvicorn на том же порту.
"""
import sys, os, importlib, time, socket

PID_FILE = '/tmp/control_api.pid'
HOST = '0.0.0.0'
PORT = 8765

def is_port_open(host, port):
    """Проверить, занят ли порт"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.close()
        return False
    except OSError:
        return True

def reload_dashboard():
    """Принудительно перезагрузить control_api"""
    print(f"🔄 Reload dashboard on {HOST}:{PORT}")
    
    # 1. Убьём старый процесс на порту (если запущен сам uvicorn)
    if is_port_open(HOST, PORT):
        import subprocess
        result = subprocess.run(
            f"lsof -ti:{PORT} | xargs -r kill -9 2>/dev/null",
            shell=True, capture_output=True
        )
        time.sleep(1)
        print("   Старый процесс убит")
    
    # 2. Перезагрузим модуль
    if 'control_api' in sys.modules:
        del sys.modules['control_api']
        print("   Старый модуль выгружен")
    
    # 3. Импортируем новый
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import control_api
    importlib.reload(control_api)
    
    # 4. Запускаем на том же порту (в фоне)
    import threading
    t = threading.Thread(
        target=control_api.run_server,
        kwargs={'host': HOST, 'port': PORT},
        daemon=True,
        name='dashboard_reloaded'
    )
    t.start()
    
    # 5. Ждём готовности
    time.sleep(2)
    if is_port_open(HOST, PORT):
        print(f"✅ Дашборд перезагружен на http://localhost:{PORT}")
    else:
        print(f"❌ Не удалось запустить дашборд на порту {PORT}")
    
    # 6. Сохраняем PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

if __name__ == '__main__':
    reload_dashboard()
