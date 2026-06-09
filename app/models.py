from datetime import datetime
from . import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from itsdangerous import URLSafeTimedSerializer as Serializer
from flask import current_app

# The User model represents a user of the application.
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    full_name = db.Column(db.String(128))
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(64), default='user')
    email_confirmed = db.Column(db.Boolean, default=False)
    email_confirm_token = db.Column(db.String(128))
    # This relationship allows us to access the user's prediction history.
    predictions = db.relationship('PredictionHistory', backref='user', lazy='dynamic')
    # This relationship allows us to access the user's uploaded documents.
    documents = db.relationship('Document', backref='user', lazy='dynamic')

    # Hashes the user's password and stores it in the password_hash field.
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # Checks if the given password matches the stored password hash.
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Generates a confirmation token for email verification.
    def generate_confirmation_token(self, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'])
        self.email_confirm_token = s.dumps({'confirm': self.id})
        return self.email_confirm_token

    # Confirms a user's email address using a confirmation token.
    def confirm(self, token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token, max_age=3600)
        except:
            return False
        if data.get('confirm') != self.id:
            return False
        self.email_confirmed = True
        db.session.add(self)
        return True

    # Generates a password reset token.
    def generate_reset_token(self, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'])
        return s.dumps({'reset': self.id})

    # Verifies a password reset token.
    @staticmethod
    def verify_reset_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return None
        return User.query.get(data.get('reset'))

    def __repr__(self):
        return f'<User {self.username}>'

# The PredictionHistory model stores the prediction history for each user.
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Store input data as a JSON string for flexibility.
    input_data = db.Column(db.String(1024))
    prediction = db.Column(db.Boolean)
    probability = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __repr__(self):
        return f'<PredictionHistory {self.id}>'

# The Document model stores information about uploaded documents.
class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(128))
    filepath = db.Column(db.String(256))
    upload_date = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __repr__(self):
        return f'<Document {self.filename}>'