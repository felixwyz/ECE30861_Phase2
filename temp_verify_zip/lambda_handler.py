from mangum import Mangum
from api.autograder_routes import app

handler = Mangum(app)
