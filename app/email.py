import os

def send_email(to, subject, template):
    print(f"Sending email to {to}")
    print(f"Subject: {subject}")
    print(f"Body: {template}")
    # Create a directory for emails if it doesn't exist
    if not os.path.exists('emails'):
        os.makedirs('emails')
    # Save the email to a file
    with open(f"emails/{to}.txt", "w") as f:
        f.write(f"Subject: {subject}\n\n{template}")
