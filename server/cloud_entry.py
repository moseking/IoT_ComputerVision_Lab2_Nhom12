"""Container entry point; PORT is supplied by Render."""
import os
import uvicorn
if __name__=='__main__':
    port=int(os.getenv('PORT','10000'))
    if not 1<=port<=65535:raise ValueError('Invalid PORT')
    uvicorn.run('server.cloud_app:app',host='0.0.0.0',port=port,workers=1)
