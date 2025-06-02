from flask import Flask, render_template, request, redirect, session, jsonify

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Used for session management

@app.route("/")
def home():
    if "user_name" in session:
        return redirect("/dashboard")
    return render_template("index.html")

@app.route("/set_user", methods=["GET"])
def set_user():
    user_name = request.args.get("name")
    user_image = request.args.get("image", "default.jpg")

    if user_name:
        session["user_name"] = user_name
        session["user_image"] = user_image
        return jsonify({"message": "User session set!"}), 200
    return jsonify({"error": "Failed to set user session"}), 400

@app.route("/get_user")
def get_user():
    if "user_name" in session:
        return jsonify({
            "name": session["user_name"],
            "image": session.get("user_image", "default.jpg")
        })
    return jsonify({"error": "User not logged in"}), 401

@app.route("/dashboard")
def dashboard():
    if "user_name" in session:
        return render_template("dashboard.html", user=session["user_name"])
    return redirect("/")

@app.route("/logout")
def logout():
    session.pop("user_name", None)
    session.pop("user_image", None)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
