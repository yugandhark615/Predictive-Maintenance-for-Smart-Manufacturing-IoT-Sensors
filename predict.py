import joblib, numpy as np
m=joblib.load("model.pkl"); model=m["binary_model"]; type_model=m["type_model"]
x=np.array([[float(input("Temperature (K): ")),float(input("Rotational Speed (RPM): ")),float(input("Torque (Nm): ")),float(input("Tool Wear (min): "))]])
fail=int(model.predict(x)[0]); print(f"Failure probability: {model.predict_proba(x)[0][1]*100:.2f}%")
print("Failure:","YES" if fail else "NO"); print("Failure type:",type_model.predict(x)[0] if fail else "No Failure")
