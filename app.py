import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Rockfall Risk Assessment",
    page_icon="ROCK",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "rockfall_synthetic_data.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "rainfall_mm_past_24h",
    "seismic_activity",
    "joint_water_pressure_kPa",
    "vibration_level",
    "displacement_mm"
]

FEATURE_LABELS = {
    "rainfall_mm_past_24h": "Rainfall (mm / last 24h)",
    "seismic_activity": "Seismic Activity (Magnitude)",
    "joint_water_pressure_kPa": "Joint Water Pressure (kPa)",
    "vibration_level": "Vibration Level (unitless)",
    "displacement_mm": "Slope Displacement (mm)"
}

COLUMN_ALIASES = {
    "rainfall": "rainfall_mm_past_24h",
    "rainfall_mm": "rainfall_mm_past_24h",
    "rainfall_mm_past_24h": "rainfall_mm_past_24h",
    "rainfalllast24h": "rainfall_mm_past_24h",
    "rainfall_mm_last_24h": "rainfall_mm_past_24h",
    "rainfallmm24h": "rainfall_mm_past_24h",
    "seismic": "seismic_activity",
    "seismic_activity": "seismic_activity",
    "seismicactivity": "seismic_activity",
    "seismic_magnitude": "seismic_activity",
    "seismicactivitymagnitude": "seismic_activity",
    "joint_water_pressure": "joint_water_pressure_kPa",
    "joint_water_pressure_kpa": "joint_water_pressure_kPa",
    "jointwaterpressure": "joint_water_pressure_kPa",
    "water_pressure": "joint_water_pressure_kPa",
    "waterpressure": "joint_water_pressure_kPa",
    "pore_pressure": "joint_water_pressure_kPa",
    "joint_water_pressure_kpa_": "joint_water_pressure_kPa",
    "vibration": "vibration_level",
    "vibration_level": "vibration_level",
    "vibrationlevel": "vibration_level",
    "vibration_sensor": "vibration_level",
    "vibrationsensor": "vibration_level",
    "groundvibration": "vibration_level",
    "displacement": "displacement_mm",
    "displacement_mm": "displacement_mm",
    "ground_displacement": "displacement_mm",
    "grounddisplacement": "displacement_mm",
    "slope_displacement": "displacement_mm",
    "slope_displacement_mm": "displacement_mm"
}

FEATURE_STEPS = {
    "rainfall_mm_past_24h": 0.1,
    "seismic_activity": 0.05,
    "joint_water_pressure_kPa": 0.5,
    "vibration_level": 0.01,
    "displacement_mm": 0.1
}

RISK_ORDER = ["Low", "Medium", "High", "Critical"]

RISK_BADGES = {
    "Low": ("#1E8449", "#F4FCF7"),
    "Medium": ("#9A7D0A", "#FFF9E6"),
    "High": ("#D35400", "#FFF2E6"),
    "Critical": ("#8B0000", "#FFECEA")
}

RECOMMENDATIONS = {
    "Low": [
        "Continue routine operations",
        "Maintain standard monitoring cadence",
        "Log sensor readings for traceability"
    ],
    "Medium": [
        "Increase inspection frequency",
        "Notify geotechnical supervisor",
        "Restrict non-essential access near the slope"
    ],
    "High": [
        "Pause excavation activities in the zone",
        "Position emergency response crew on standby",
        "Expand real-time monitoring coverage"
    ],
    "Critical": [
        "Initiate immediate evacuation",
        "Activate emergency command protocol",
        "Secure perimeter and halt all site activity"
    ]
}

SAMPLE_SCENARIOS = {
    "Stable bench (Low risk)": {
        "rainfall_mm_past_24h": 2.0,
        "seismic_activity": 0.9,
        "joint_water_pressure_kPa": 32.0,
        "vibration_level": 0.25,
        "displacement_mm": 6.0
    },
    "Softening slope (Medium risk)": {
        "rainfall_mm_past_24h": 8.5,
        "seismic_activity": 2.4,
        "joint_water_pressure_kPa": 39.0,
        "vibration_level": 0.5,
        "displacement_mm": 15.5
    },
    "Tension crack detected (High risk)": {
        "rainfall_mm_past_24h": 10.5,
        "seismic_activity": 3.5,
        "joint_water_pressure_kPa": 41.0,
        "vibration_level": 0.8,
        "displacement_mm": 24.0
    },
    "Impending failure (Critical risk)": {
        "rainfall_mm_past_24h": 11.5,
        "seismic_activity": 4.5,
        "joint_water_pressure_kPa": 42.0,
        "vibration_level": 1.15,
        "displacement_mm": 31.0
    }
}


def _normalize_column_key(name: str) -> str:
    """Return a simplified key for matching user-provided column headers."""
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


@st.cache_resource
def load_model_assets():
    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def compute_feature_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in FEATURE_COLUMNS:
        series = df[col]
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "p10": float(series.quantile(0.10)),
            "p90": float(series.quantile(0.90))
        }
    return stats


def align_feature_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    rename_map = {}
    normalized_aliases = {_normalize_column_key(k): v for k, v in COLUMN_ALIASES.items()}
    for col in df.columns:
        key = _normalize_column_key(col)
        if key in normalized_aliases:
            rename_map[col] = normalized_aliases[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    try:
        df = df[FEATURE_COLUMNS].astype(float)
    except ValueError as exc:
        raise ValueError("All feature columns must be numeric.") from exc
    return df


def predict_dataframe(
    input_df: pd.DataFrame,
    model,
    scaler,
    label_encoder
):
    features = align_feature_columns(input_df)
    scaled = scaler.transform(features)
    predictions = model.predict(scaled)
    try:
        labels = label_encoder.inverse_transform(predictions.astype(int))
    except Exception:
        labels = predictions
    probabilities = None
    probability_labels = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(scaled)
            classes = getattr(model, "classes_", np.arange(probabilities.shape[1]))
            try:
                probability_labels = label_encoder.inverse_transform(classes.astype(int))
            except Exception:
                probability_labels = [str(c) for c in classes]
        except Exception:
            probabilities = None
            probability_labels = None
    return features, np.array(labels), probabilities, probability_labels


def render_risk_badge(label: str):
    color, text_color = RISK_BADGES.get(label, ("#34495E", "#FFFFFF"))
    st.markdown(
        f"<div style='padding:0.85rem;border-radius:0.6rem;background:{color};"
        f"color:{text_color};text-align:center;font-weight:600;font-size:1.2rem;'>"
        f"Predicted risk: {label}</div>",
        unsafe_allow_html=True
    )


def render_probability_chart(labels, probabilities):
    df = pd.DataFrame({
        "Risk Level": [str(label) for label in labels],
        "Probability": (probabilities * 100).round(2)
    })
    order_map = {level: idx for idx, level in enumerate(RISK_ORDER)}
    df["_order"] = df["Risk Level"].map(order_map).fillna(len(RISK_ORDER))
    df = df.sort_values("_order").drop(columns="_order")
    fig = px.bar(
        df,
        x="Risk Level",
        y="Probability",
        color="Probability",
        color_continuous_scale="Blues",
        range_y=[0, 100],
        labels={"Probability": "Probability (%)"}
    )
    fig.update_layout(showlegend=False, height=360, margin=dict(l=24, r=24, t=24, b=24))
    st.plotly_chart(fig, use_container_width=True)


def get_test_set(df: pd.DataFrame, label_encoder):
    X = df[FEATURE_COLUMNS]
    y = label_encoder.transform(df["risk_level"])
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def compute_evaluation(df: pd.DataFrame, model, scaler, label_encoder):
    X_train, X_test, y_train, y_test = get_test_set(df, label_encoder)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    classes = label_encoder.classes_.tolist()
    report = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    return X_test, y_test, y_pred, classes, report, cm


def render_overview(df: pd.DataFrame, feature_stats: dict):
    st.header("Project Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", f"{len(df):,}")
    col2.metric("Features", str(len(FEATURE_COLUMNS)))
    col3.metric("Risk classes", str(len(RISK_ORDER)))

    st.markdown(
        "The dataset combines synthetic sensor readings with real-world statistical"
        " drivers to model open-pit slope stability. Risk labels follow the logical"
        " thresholds authored in Notebook 1."
    )

    risk_counts = df["risk_level"].value_counts().reindex(RISK_ORDER)
    risk_pct = (risk_counts / risk_counts.sum() * 100).round(2)

    st.subheader("Risk distribution")
    fig = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        text=[f"{p}%" for p in risk_pct.fillna(0)],
        labels={"x": "Risk level", "y": "Samples"},
        color=risk_counts.index,
        color_discrete_sequence=[RISK_BADGES[level][0] for level in RISK_ORDER]
    )
    fig.update_layout(showlegend=False, height=380, margin=dict(l=24, r=24, t=24, b=24))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature ranges (10th-90th percentile)")
    ranges = []
    for col in FEATURE_COLUMNS:
        stats = feature_stats[col]
        ranges.append({
            "Feature": FEATURE_LABELS[col],
            "Typical range": f"{stats['p10']:.2f} - {stats['p90']:.2f}",
            "Absolute min": f"{stats['min']:.2f}",
            "Absolute max": f"{stats['max']:.2f}"
        })
    st.dataframe(pd.DataFrame(ranges))

    st.subheader("Workflow recap")
    st.markdown(
        "- Notebook 1 synthesised 20,000 labelled slope events with statistically realistic drivers.\n"
        "- Notebook 2 validated feature behaviour and highlighted displacement dominance.\n"
        "- Notebook 3 trained multiple models; the XGBClassifier delivered 99% recall on Critical events.\n"
        "- Notebook 4 confirmed deployment readiness via confusion matrix and feature importance."
    )


def render_manual_input(feature_stats: dict) -> pd.DataFrame:
    st.subheader("Manual input")
    st.info(
        "Field heuristics: displacement above 22 mm usually elevates risk to High,"
        " while readings beyond 30 mm drive the Critical rules authored in Notebook 1."
    )
    cols = st.columns(2)
    values = {}
    for idx, feature in enumerate(FEATURE_COLUMNS):
        stats = feature_stats[feature]
        step = FEATURE_STEPS[feature]
        default = float(np.clip(stats["median"], stats["min"], stats["max"]))
        widget = cols[idx % 2]
        with widget:
            values[feature] = st.slider(
                FEATURE_LABELS[feature],
                min_value=float(stats["min"]),
                max_value=float(stats["max"]),
                value=default,
                step=step
            )
            st.caption(
                f"Typical training range: {stats['p10']:.2f} - {stats['p90']:.2f}"
            )
            if values[feature] < stats["p10"] or values[feature] > stats["p90"]:
                st.markdown(
                    "<span style='color:#F4D03F;'>Outside typical range observed during training.</span>",
                    unsafe_allow_html=True
                )
    return pd.DataFrame([values])


def render_scenarios():
    st.subheader("Scenario explorer")
    scenario = st.selectbox("Choose scenario", list(SAMPLE_SCENARIOS.keys()))
    scenario_df = pd.DataFrame([SAMPLE_SCENARIOS[scenario]])
    st.write("Sensor profile")
    st.dataframe(scenario_df)
    return scenario_df


def render_batch_section(model, scaler, label_encoder):
    st.subheader("Batch predictions")
    st.info("Upload CSV with columns matching the feature set. Extra columns will be ignored.")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if not uploaded:
        return
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        return
    st.write("Preview")
    st.dataframe(raw_df.head())
    if st.button("Run batch prediction", type="primary"):
        try:
            _, labels, probabilities, _ = predict_dataframe(raw_df, model, scaler, label_encoder)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return
        results = raw_df.copy()
        results["Predicted_Risk"] = labels
        if probabilities is not None:
            confidence = probabilities.max(axis=1) * 100
            results["Confidence_%"] = confidence.round(2)
        st.success("Predictions completed")
        st.dataframe(results)
        st.download_button(
            label="Download results",
            data=results.to_csv(index=False),
            file_name="rockfall_predictions.csv",
            mime="text/csv"
        )


def render_recommendations(label: str):
    items = RECOMMENDATIONS.get(label, [])
    if not items:
        return
    for item in items:
        st.write(f"- {item}")


def render_prediction_page(df, model, scaler, label_encoder, feature_stats):
    st.header("Predict rockfall risk")
    st.markdown("The model mirrors the preprocessing pipeline defined in Notebook 3 (label encoder + StandardScaler + XGBClassifier).")
    tabs = st.tabs(["Manual input", "Scenario explorer", "Batch predictions"])

    with tabs[0]:
        manual_df = render_manual_input(feature_stats)
        if st.button("Predict", key="manual_predict", type="primary"):
            try:
                _, labels, probabilities, prob_labels = predict_dataframe(manual_df, model, scaler, label_encoder)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                label = labels[0]
                render_risk_badge(label)
                st.subheader("Recommended response")
                render_recommendations(label)
                if probabilities is not None and prob_labels is not None:
                    render_probability_chart(prob_labels, probabilities[0])
                st.subheader("Input summary")
                st.dataframe(manual_df)

    with tabs[1]:
        scenario_df = render_scenarios()
        if st.button("Predict", key="scenario_predict", type="primary"):
            try:
                _, labels, probabilities, prob_labels = predict_dataframe(scenario_df, model, scaler, label_encoder)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                label = labels[0]
                render_risk_badge(label)
                st.subheader("Recommended response")
                render_recommendations(label)
                if probabilities is not None and prob_labels is not None:
                    render_probability_chart(prob_labels, probabilities[0])

    with tabs[2]:
        render_batch_section(model, scaler, label_encoder)


def render_model_diagnostics(df, model, scaler, label_encoder):
    st.header("Model diagnostics")
    X_test, y_test, y_pred, classes, report, cm = compute_evaluation(df, model, scaler, label_encoder)

    accuracy = report.get("accuracy", 0) * 100
    critical_recall = report.get("Critical", {}).get("recall", 0) * 100
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.2f}%")
    col2.metric("Critical recall", f"{critical_recall:.2f}%")
    col3.metric("Macro F1", f"{macro_f1:.2f}%")

    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    st.subheader("Confusion matrix (test split, stratified)")
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_df.values,
            x=cm_df.columns,
            y=cm_df.index,
            text=cm_df.values,
            texttemplate="%{text}",
            colorscale="Blues"
        )
    )
    fig.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification report")
    report_df = pd.DataFrame(report).transpose()
    display_df = report_df.loc[classes, ["precision", "recall", "f1-score", "support"]]
    st.dataframe(display_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}))

    if hasattr(model, "feature_importances_"):
        st.subheader("Feature importance (XGBoost gain)")
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig_imp = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            labels={"Importance": "Importance score", "Feature": "Feature"},
            color="Importance",
            color_continuous_scale="viridis"
        )
        fig_imp.update_layout(showlegend=False, height=380, margin=dict(l=40, r=40, t=24, b=24))
        st.plotly_chart(fig_imp, use_container_width=True)


def render_about_section():
    st.header("Project context")
    st.markdown(
        "This Streamlit interface deploys the pipeline delivered in the four-course notebooks."
        " It relies on the synthetic dataset engineered from Kaggle rainfall and seismic"
        " distributions and exposes the champion XGBClassifier selected in Notebook 3." 
    )
    st.markdown(
        "- **Data source:** `data/rockfall_synthetic_data.csv` (20,000 samples).\n"
        "- **Preprocessing:** StandardScaler + LabelEncoder persisted in `models/`.\n"
        "- **Model:** `models/xgb_model.pkl` with multi-class objective.\n"
        "- **Risk labels:** Low, Medium, High, Critical (imbalanced by design)."
    )
    st.markdown(
        "Run notebooks sequentially if you need to regenerate assets or revisit analysis:"
        " [`01` sourcing -> `02` EDA -> `03` modelling -> `04` interpretation]."
    )


def main():
    st.title("Open-Pit Mine Rockfall Risk Assessment")
    try:
        model, scaler, label_encoder = load_model_assets()
    except FileNotFoundError as exc:
        st.error(f"Model asset missing: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load model assets: {exc}")
        st.stop()

    try:
        dataset = load_dataset()
    except FileNotFoundError:
        st.error("Dataset not found. Run Notebook 1 to generate `rockfall_synthetic_data.csv`.")
        st.stop()
    feature_stats = compute_feature_stats(dataset)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Predict", "Diagnostics", "About"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model artefacts")
    st.sidebar.write("xgb_model.pkl")
    st.sidebar.write("scaler.pkl")
    st.sidebar.write("label_encoder.pkl")

    if page == "Overview":
        render_overview(dataset, feature_stats)
    elif page == "Predict":
        render_prediction_page(dataset, model, scaler, label_encoder, feature_stats)
    elif page == "Diagnostics":
        render_model_diagnostics(dataset, model, scaler, label_encoder)
    else:
        render_about_section()


if __name__ == "__main__":
    main()
