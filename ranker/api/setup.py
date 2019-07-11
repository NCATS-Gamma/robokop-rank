'''
Set up Flask server
'''

import logging
import logging.config

from flask import Flask, Blueprint, request
from flask_restful import Api
from flasgger import Swagger, LazyJSONEncoder, LazyStrings
from flask_cors import CORS

from ranker.api.logging_config import setup_main_logger


setup_main_logger()
logger = logging.getLogger("ranker")

app = Flask(__name__, static_folder='../pack', template_folder='../templates')
# Set default static folder to point to parent static folder where all
# static assets can be stored and linked
# app.config.from_pyfile('robokop_flask_config.py')

api_blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(api_blueprint)
CORS(app)
app.register_blueprint(api_blueprint)

template = {
    "openapi": "3.0.1",
    "info": {
        "title": "ROBOKOP Ranker",
        "description": "An API for answering biomedical questions",
        "contact": {
            "responsibleOrganization": "CoVar Applied Technologies",
            "responsibleDeveloper": "patrick@covar.com",
            "email": "patrick@covar.com",
            "url": "www.covar.com",
        },
        "termsOfService": "<url>",
        "version": "0.0.1"
    },
    "schemes": [
        "http",
        "https"
    ],
    'swaggerUiPrefix': LazyString (lambda : request.environ.get('X-Swagger-Prefix', ''))
}
app.json_encoder = LazyJSONEncoder
app.config['SWAGGER'] = {
    'title': 'ROBOKOP Ranker API',
    'uiversion': 3
}
swagger_config = {
    "headers": [
    ],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/ranker/spec',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "swagger_ui": True,
    "specs_route": "/apidocs/",
    "openapi": "3.0.1"
}
swagger = Swagger(app, template=template, config=swagger_config)

# Should be catching werkzeug.exceptions.InternalServerError instead?
@app.errorhandler(Exception)
def handle_error(ex):
    logger.exception(ex)
    return "Internal server error. See the logs for details.", 500
app.config['PROPAGATE_EXCEPTIONS'] = True
app.url_map.strict_slashes = False
