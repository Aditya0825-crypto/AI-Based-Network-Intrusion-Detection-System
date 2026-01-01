import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------
# DATA GENERATION (RULE-BASED)
# ------------------------------------
def load_data():
    np.random.seed(42)

    packet_size = np.random.randint(20, 1500, 1000)
    duration = np.random.rand(1000) * 10
    src_bytes = np.random.randint(0, 10000, 1000)
    dst_bytes = np.random.randint(0, 10000, 1000)
    protocol = np.random.choice([0, 1, 2], 1000)

    label = []
    for i in range(1000):
        # Attack rules (simulated)
        if packet_size[i] > 1000 and src_bytes[i] > 7000:
            label.append(1)  # Attack
        else:
            label.append(0)  # Normal

    df = pd.DataFrame({
        "packet_size": packet_size,
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "protocol": protocol,
        "label": label
    })

    return df

# ------------------------------------
# STREAMLIT UI CONFIG
# ------------------------------------
st.set_page_config(page_title="AI-Based NIDS", layout="centered")

st.title("🔐 AI-Based Network Intrusion Detection System")
st.write(
    "This AI-powered system detects malicious network traffic using "
    "a Random Forest Machine Learning model."
)

# ------------------------------------
# LOAD DATA
# ------------------------------------
df = load_data()

st.subheader("📊 Sample Network Traffic Data")
st.dataframe(df.head())

# ------------------------------------
# SIDEBAR - MODEL TRAINING
# ------------------------------------
st.sidebar.header("⚙️ Model Controls")

if st.sidebar.button("Train Model Now"):
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success("✅ Model Trained Successfully!")
    st.write(f"🎯 **Accuracy:** {acc * 100:.2f}%")

    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📈 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.session_state["model"] = model

# ------------------------------------
# LIVE TRAFFIC SIMULATOR
# ------------------------------------
st.subheader("🚦 Live Traffic Simulator")

packet_size = st.number_input("Packet Size", 20, 1500, 800)
duration = st.number_input("Duration (seconds)", 0.0, 10.0, 2.0)
src_bytes = st.number_input("Source Bytes", 0, 10000, 5000)
dst_bytes = st.number_input("Destination Bytes", 0, 10000, 3000)
protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])

protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}
protocol_value = protocol_map[protocol]

if st.button("Detect Intrusion"):
    if "model" not in st.session_state:
        st.warning("⚠️ Please train the model first.")
    else:
        input_data = np.array([
            [packet_size, duration, src_bytes, dst_bytes, protocol_value]
        ])

        prediction = st.session_state["model"].predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 Intrusion Detected (Malicious Traffic)")
        else:
            st.success("✅ Normal Network Traffic Detected")

# ------------------------------------
# FOOTER
# ------------------------------------
st.markdown("---")
st.markdown("**Developed as an Academic AI-Based Network Security Project**")
