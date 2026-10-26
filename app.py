import streamlit as st
import matplotlib.pyplot as plt

from src.predict import predict_failure, THRESHOLD

st.set_page_config(
    page_title="Jenkins Build Failure Predictor",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Jenkins Build Failure Predictor")
st.caption("Pre-build CI/CD risk estimation using historical signals")

st.sidebar.header("⚙️ Pre-build Parameters")

code_changes = st.sidebar.slider(
    "🧩 Lines Changed",
    min_value=1,
    max_value=1200,
    value=300
)

hour = st.sidebar.slider(
    "🕒 Planned Build Hour",
    min_value=0,
    max_value=23,
    value=14
)

prev_failure = st.sidebar.radio(
    "❗ Did the Previous Build Fail?",
    options=["No", "Yes"]
)
prev_failure = int(prev_failure == "Yes")

failure_last_5 = st.sidebar.slider(
    "📉 Failures in the Previous 5 Builds",
    min_value=0,
    max_value=5,
    value=1
)

if st.sidebar.button("🔍 Analyze Build Risk", type="primary"):
    features = {
        "code_changes": code_changes,
        "hour": hour,
        "prev_failure": prev_failure,
        "failure_last_5": failure_last_5
    }

    try:
        ml_risk = predict_failure(features)

        col1, col2 = st.columns(2)
        col1.metric("📊 Predicted Failure Probability", f"{ml_risk:.1%}")
        col2.metric("⚙️ Warning Threshold", f"{THRESHOLD:.0%}")

        st.progress(max(0.0, min(ml_risk, 1.0)))

        if ml_risk >= THRESHOLD:
            st.warning(
                "🟠 Elevated predicted risk — review the build context."
            )
        else:
            st.success(
                "🟢 Below the configured warning threshold."
            )

        st.caption(
            "This is an experimental prediction based on synthetic data. "
            "It does not guarantee build success or failure."
        )

        st.subheader("📈 Prediction Overview")
        fig, ax = plt.subplots()
        ax.bar(
            ["Predicted Failure Probability", "Warning Threshold"],
            [ml_risk, THRESHOLD]
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability / threshold")
        st.pyplot(fig)
        plt.close(fig)

    except (ValueError, TypeError, FileNotFoundError) as exc:
        st.error(f"Could not analyze this build: {exc}")

st.markdown("---")
st.caption("Jenkins Build Failure Prediction | Experimental ML prototype")
