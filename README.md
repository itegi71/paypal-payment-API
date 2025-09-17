This project is a Python-based PayPal Sandbox integration that demonstrates how to create and process payments programmatically.
It provides a simple flow where a payment request is made, authenticated against PayPal’s Sandbox environment, and then executed.
The uniqueness of this project is in its minimal yet extendable design.
Built with Python 3 and tested in Kali Linux / Sublime Text / VSCode.Implements logging & retries to handle failed transactions gracefully.
Can be adapted to automation tasks or extended into a Django/Flask web app.

⚙️ How It Works

The script authenticates with PayPal Sandbox API using your Client ID and Secret Key.
An access token is generated.
The script attempts to create a test payment (default: $1.00).
If authentication fails, the script retries up to 3 times with detailed logs.
Successful payments return a transaction response from PayPal.
