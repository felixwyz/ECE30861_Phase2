from mangum import Mangum
from api.routes import app

handler = Mangum(app)
