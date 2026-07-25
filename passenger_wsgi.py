import sys
import os
import gc

# Batasi penggunaan multi-threading bawaan C-Libraries untuk hemat RAM (Penting untuk Shared Hosting)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add the application directory to the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import ASGIMiddleware to convert FastAPI (ASGI) to WSGI
from a2wsgi import ASGIMiddleware
from app import app as fastapi_app

# Expose 'application' for Phusion Passenger in cPanel
_application = ASGIMiddleware(fastapi_app)

def application(environ, start_response):
    # Panggil garbage collector pada setiap request untuk membersihkan sisa memory
    gc.collect()
    return _application(environ, start_response)
