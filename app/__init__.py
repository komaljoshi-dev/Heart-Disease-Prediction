import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .config import Config
from flask_login import LoginManager

# Create extension instances
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
# The login_view is the name of the view to redirect to when the user needs to log in.
login_manager.login_view = 'main.login'

# The user_loader function is used by Flask-Login to load a user from the database.
@login_manager.user_loader
def load_user(user_id):
    from .models import User
    return User.query.get(int(user_id))

# The create_app function is the application factory.
def create_app(config_class=Config):
    """Creates and configures the Flask application."""
    # Create the Flask app instance.
    # instance_relative_config=True tells the app that configuration files are relative to the instance folder.
    app = Flask(__name__, instance_relative_config=True)

    # Load configuration from the config object.
    app.config.from_object(config_class)

    # Ensure the instance folder exists.
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize extensions with the app.
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    with app.app_context():
        # Import models to ensure they are registered with SQLAlchemy.
        from . import models

        # Create database tables for our models (if they don't exist).
        db.create_all()

    # Import and register blueprints.
    from .views import main_bp
    app.register_blueprint(main_bp)

    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    @app.context_processor
    def inject_current_year():
        from datetime import datetime
        return {'current_year': datetime.now().year}

    return app
