# This file is the main entry point for the application.

from app import create_app, db
from app.models import User, PredictionHistory

# Create the Flask app instance using the application factory
app = create_app()

@app.shell_context_processor 
def make_shell_context():
    """Creates a shell context that adds the database and models to the shell session."""
    return {'db': db, 'User': User, 'PredictionHistory': PredictionHistory}

if __name__ == '__main__':
    # Note: debug=False is recommended for production
    app.run(debug=True)
 