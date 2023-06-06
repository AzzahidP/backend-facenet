from app import db
from datetime import datetime

class FacenetUser(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    full_name = db.Column(db.String(250), nullable=False)
    vector = db.Column(db.String(7500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __represent__(self):
        return '<FacenetUser {}>'.format(self.name)