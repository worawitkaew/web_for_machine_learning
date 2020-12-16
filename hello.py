from flask import Flask,render_template
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

# ทำตามจาก วิดีโอ
@app.route("/show")
def show():
    name = "Diamond"
    return render_template("index.html",name=name)

# แสดงhtmlตัวอย่างจากในเว็ป flask
@app.route("/base")
def base():
    return render_template("base.html")

@app.route("/registe")
def register():
    return render_template("auth/register.html")

@app.route("/logi")
def login():
    return render_template("auth/login.html")

if __name__ == "__main__":
    app.run(debug=True)