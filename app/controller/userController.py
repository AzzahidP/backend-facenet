from app import db, app
from flask import jsonify
from numpy import linalg
import numpy as np

def save_to_db(Model, full_name, vector):
    try:
        new_user = Model(full_name=full_name, vector=vector)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'status': 'ok', 'message': 'Registrasi Berhasil'})
    except Exception as e:
        return jsonify({'status': 'failed', 'message': {str(e)}})
    
def verify_from_db(Model, min_dist, vector_input):
    db = Model.query.with_entities(Model.full_name,Model.vector)
    identity = 'unknown09123' 
    for rows in db:
        vector_db = rows.vector
        vector_db = np.fromstring(vector_db[2:-2], sep=' ') 
        dist = linalg.norm(vector_input - vector_db)
        if dist < min_dist:
            min_dist = dist
            identity = rows.full_name 
    
    return identity

def get_all_details(Model, key):
    query = Model.query.filter_by(full_name=key).first()
    return query