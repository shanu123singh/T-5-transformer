def create_input(row,question):
     return f"""
soil_nitrogen: {row['N']}
soil_phosphorus: {row['P']}
soil_potassium: {row['K']}
temperature: {row['temperature']}
humidity: {row['humidity']}
ph: {row['ph']}
rainfall: {row['rainfall']}
question: {question}
"""
