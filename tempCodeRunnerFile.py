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
    
    # Resh