import socket
import uvicorn
import webbrowser

if __name__ == "__main__":
    host = socket.gethostbyname(socket.gethostname())
    web_url = f"http://{host}:8000/docs"  # URL của trang tài liệu FastAPI
    webbrowser.open(web_url)  # Tự động mở trình duyệt với URL

    uvicorn.run("app:app", host=host, port=8000, reload=True)