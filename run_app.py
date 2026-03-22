import asyncio
import sys
import threading
from streamlit.web import cli as stcli

# More aggressive patch for Python 3.14 compatibility
def patch_asyncio():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Override get_event_loop to be more permissive
    _get_loop = asyncio.get_event_loop
    def robust_get_loop():
        try:
            return _get_loop()
        except RuntimeError:
            # If we're in the main thread, return the loop we created
            if threading.current_thread() is threading.main_thread():
                return loop
            # Otherwise, create a new one for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop
            
    asyncio.get_event_loop = robust_get_loop

if __name__ == "__main__":
    patch_asyncio()
    sys.argv = ["streamlit", "run", "app/streamlit_app.py"]
    sys.exit(stcli.main())
