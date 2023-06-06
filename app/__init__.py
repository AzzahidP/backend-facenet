from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from dbconfig import AppConfig

app = Flask(__name__)
cors = CORS(app)
app.config.from_object(AppConfig)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

from app.dbmodel import ageitgeyUser, facenetUser
from app import routes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001')