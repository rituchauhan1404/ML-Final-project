from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import pickle
import sqlite3
import numpy as np


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///majordb.db'
db = SQLAlchemy(app)

class Cars(db.Model):
    car_name = db.Column(db.Text, primary_key=True)
    reviews_count = db.Column(db.Integer)
    fuel_type = db.Column(db.Text)
    engine_displacement = db.Column(db.Float)
    no_cylinder = db.Column(db.Integer)
    seating_capacity = db.Column(db.Integer)
    transmission_type = db.Column(db.Text)
    fuel_tank_capacity = db.Column(db.Float)
    body_type = db.Column(db.Text)
    rating = db.Column(db.Float)
    starting_price = db.Column(db.Integer)
    ending_price = db.Column(db.Integer)
    max_torque_nm = db.Column(db.Float)
    max_torque_rpm = db.Column(db.Integer)
    max_power_bhp = db.Column(db.Float)
    max_power_rp = db.Column(db.Integer)
    brand = db.Column(db.Text)
    average_price = db.Column(db.Float)
    fuel_efficiency = db.Column(db.Float)

class Satellite(db.Model):
    Discipline = db.Column(db.Text)
    Launch_mass = db.Column(db.Float)
    Launch_date = db.Column(db.TIMESTAMP, primary_key=True)
    Launch_vehicle = db.Column(db.Text)
    Launch_site = db.Column(db.Text)
    Periapsis_km = db.Column(db.Float)
    Apoapsis_km = db.Column(db.Float)
    Period_in_minutes = db.Column(db.Float)
    Success = db.Column(db.Text)
    Orbit_Type = db.Column(db.Text)
    Launch_year = db.Column(db.Integer)
    Launch_month = db.Column(db.Integer)
    Satellite_Purpose = db.Column(db.Text)


d1_classification_model = pickle.load(open("ds1_clasi.pkl", "rb"))
d1_regression_model = pickle.load(open("ds1_reg.pkl", "rb"))
d2_classification_model = pickle.load(open("ds2_classi.pkl", "rb"))
d2_regression_model = pickle.load(open("ds2_reg.pkl", "rb"))

def get_data(table_name):
    return table_name.query.order_by(db.func.random()).limit(3).all()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classification")
def classification():
    return render_template("d1m1.html", pred_txt="", random_cars=get_data(Cars))

@app.route("/d2_classification")
def d2_classification():
    return render_template("d2m1.html", random_sat=get_data(Satellite), pred_txt ="")

@app.route("/regression")
def regression():
    return render_template("d1m2.html", pred_txt="", random_cars=get_data(Cars))

@app.route("/d2_regression")
def d2_regression():
    return render_template("d2m2.html", random_sat=get_data(Satellite), pred_txt="")

@app.route("/d2_classi", methods=["POST"])
def d2_predict():
    features_list = [
        float(request.form.get("Launch_mass")),
        float(request.form.get("Periapsis_km")),
        float(request.form.get("Apoapsis_km")),
        float(request.form.get("Period_in_minutes")),
        float(request.form.get("Orbit_Type")),
        float(request.form.get("Launch_year")),
        float(request.form.get("Launch_month")),
        float(request.form.get("Purpose"))
    ]
    
    # Reshape the features into a 2D array
    features = np.array(features_list).reshape(1, -1)

    prediction = d2_classification_model.predict(features)
    return render_template("d2m1.html", random_sat=get_data(Satellite), pred_txt="Success prediction: {}".format(prediction[0]))



@app.route("/d1_classi", methods=["POST"])
def predict():
    features_list = [
        float(request.form.get("reviews_count")),
        float(request.form.get("engine_displacement")),
        float(request.form.get("no_cylinder")),
        float(request.form.get("seating_capacity")),
        float(request.form.get("transmission_type")),
        float(request.form.get("fuel_tank_capacity")),
        float(request.form.get("body_type")),
        float(request.form.get("rating")),
        float(request.form.get("starting_price")),
        float(request.form.get("ending_price")),
        float(request.form.get("max_torque_nm")),
        float(request.form.get("max_torque_rpm")),
        float(request.form.get("max_power_bhp")),
        float(request.form.get("max_power_rp")),
        float(request.form.get("brand")),
        float(request.form.get("average_price")),
        float(request.form.get("fuel_efficiency")),
    ]

    features = np.array(features_list).reshape(1, -1)



    prediction = d1_classification_model.predict(features)
    return render_template("d1m1.html",  random_cars=get_data(Cars),pred_txt="predicted fuel type: {}".format(prediction[0]))

@app.route('/d1_reg', methods=["POST"])
def classify():
    features_list = [
        float(request.form.get("reviews_count")),
        float(request.form.get("fuel_type")),
        float(request.form.get("engine_displacement")),
        float(request.form.get("no_cylinder")),
        float(request.form.get("seating_capacity")),
        float(request.form.get("transmission_type")),
        float(request.form.get("fuel_tank_capacity")),
        float(request.form.get("body_type")),
        float(request.form.get("rating")),
        float(request.form.get("starting_price")),
        float(request.form.get("ending_price")),
        float(request.form.get("max_torque_nm")),
        float(request.form.get("max_torque_rpm")),
        float(request.form.get("max_power_bhp")),
        float(request.form.get("max_power_rp")),
        float(request.form.get("brand")),
        float(request.form.get("fuel_efficiency")),
    ]

    features = np.array(features_list).reshape(1, -1)

    prediction = d1_regression_model.predict(features)
    return render_template("d1m2.html", random_cars=get_data(Cars),pred_txt="predicted average price: {}".format(prediction[0]))


@app.route('/d2_reg' ,methods=["POST"])
def d2_reg():
    features_list = [
        float(request.form.get("Periapsis_km")),
        float(request.form.get("Apoapsis_km")),
        float(request.form.get("Period_in_minutes")),
        float(request.form.get("Orbit_Type")),
        float(request.form.get("Purpose"))
    ]

    features = np.array(features_list).reshape(1, -1)


    prediction = d2_regression_model.predict(features)
    return render_template("d2m2.html", random_sat=get_data(Satellite), pred_txt="predicted launch mass: {}".format(prediction[0]))
    


if __name__ == "__main__":
    app.run(debug=True)
