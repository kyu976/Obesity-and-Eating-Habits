import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Obesity & Eating Habits Explorer",
    page_icon="🍎",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading & preprocessing (mirrors the notebook pipeline)
# ---------------------------------------------------------------------------

@st.cache_data
def load_and_prepare():
    df = pd.read_csv("ObesityDataSet.csv")

    ordinal_mapping = {
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    }
    for col, mapping in ordinal_mapping.items():
        df[col] = df[col].map(mapping)

    obesity_order = [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ]
    obesity_mapping = {label: i for i, label in enumerate(obesity_order)}
    class_labels = list(obesity_mapping.keys())

    y = df["NObeyesdad"].map(obesity_mapping)
    X = df.drop(columns=["NObeyesdad"])
    X_encoded = pd.get_dummies(
        X, columns=["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC", "MTRANS"]
    )

    continuous_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=123, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = X_train.copy()
    X_test_sc = X_test.copy()
    X_train_sc[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_sc[continuous_cols] = scaler.transform(X_test[continuous_cols])

    rf = RandomForestClassifier(random_state=123, n_estimators=200, max_depth=None, min_samples_split=2)
    rf.fit(X_train_sc, y_train)
    y_pred = rf.predict(X_test_sc)

    feat_imp = pd.Series(rf.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
    cm = confusion_matrix(y_test, y_pred)

    return df, X_encoded, class_labels, feat_imp, cm, y_test, y_pred, rf, scaler, continuous_cols, X_train_sc


df, X_encoded, class_labels, feat_imp, cm, y_test, y_pred, model, scaler, continuous_cols, X_train_sc = load_and_prepare()

# Pretty-print label helper
def pretty(label: str) -> str:
    return label.replace("_", " ")

pretty_labels = [pretty(l) for l in class_labels]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Feature Explorer",
    "Feature Importance",
    "Confusion Matrix",
    "Predict Your Class",
])

# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Obesity & Eating Habits Explorer")
    st.markdown(
        "An interactive companion to our classification notebook. "
        "Pick a page from the sidebar to explore the dataset, model results, "
        "or predict your own obesity class."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", f"{len(df):,}")
    col2.metric("Features", X_encoded.shape[1])
    col3.metric("Classes", len(class_labels))

    st.subheader("Class Distribution")
    counts = df["NObeyesdad"].value_counts().reindex(class_labels)
    fig = px.bar(
        x=[pretty(c) for c in counts.index],
        y=counts.values,
        color=counts.values,
        color_continuous_scale="Tealgrn",
        labels={"x": "Obesity Class", "y": "Count"},
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 2. Feature Explorer
# ---------------------------------------------------------------------------
elif page == "Feature Explorer":
    st.title("Feature Explorer")
    st.markdown("Compare how lifestyle features distribute across obesity classes.")

    numeric_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "CAEC", "CALC"]

    c1, c2 = st.columns(2)
    feat_x = c1.selectbox("X-axis feature", numeric_features, index=0)
    feat_y = c2.selectbox("Y-axis feature", numeric_features, index=2)

    selected_classes = st.multiselect(
        "Filter obesity classes",
        class_labels,
        default=class_labels,
        format_func=pretty,
    )

    plot_df = df[df["NObeyesdad"].isin(selected_classes)].copy()
    plot_df["Class"] = plot_df["NObeyesdad"].apply(pretty)

    tab_scatter, tab_box = st.tabs(["Scatter", "Box Plot"])

    with tab_scatter:
        fig = px.scatter(
            plot_df, x=feat_x, y=feat_y, color="Class",
            opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Bold,
            category_orders={"Class": [pretty(c) for c in class_labels]},
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    with tab_box:
        fig = px.box(
            plot_df, x="Class", y=feat_x, color="Class",
            color_discrete_sequence=px.colors.qualitative.Bold,
            category_orders={"Class": [pretty(c) for c in class_labels]},
        )
        fig.update_layout(height=550, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Feature Importance
# ---------------------------------------------------------------------------
elif page == "Feature Importance":
    st.title("Feature Importance (Random Forest)")
    st.markdown("Which features matter most when predicting obesity class?")

    top_n = st.slider("Show top N features", 5, len(feat_imp), 15)
    top = feat_imp.head(top_n)

    fig = px.bar(
        x=top.values[::-1],
        y=top.index[::-1],
        orientation="h",
        color=top.values[::-1],
        color_continuous_scale="Sunset",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(height=max(400, top_n * 32), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Weight** and **Age** are the strongest individual predictors, but a bundle of "
        "lifestyle features (vegetable consumption, physical activity, water intake, tech use) "
        "collectively rival them — meaning behavior *does* matter for classification."
    )

# ---------------------------------------------------------------------------
# 4. Confusion Matrix
# ---------------------------------------------------------------------------
elif page == "Confusion Matrix":
    st.title("Confusion Matrix")
    st.markdown("How well does the Random Forest classify each obesity level?")

    fig = px.imshow(
        cm,
        x=pretty_labels,
        y=pretty_labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
    )
    fig.update_layout(height=600, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "The **diagonal** shows correct predictions (same class predicted as actual). "
        "Most misclassifications land on **neighboring** classes, meaning the model "
        "rarely confuses distant severity levels — it respects the ordinal structure."
    )

# ---------------------------------------------------------------------------
# 5. Predict Your Class
# ---------------------------------------------------------------------------
elif page == "Predict Your Class":
    st.title("Predict Your Obesity Class")
    st.markdown("Enter your info below and the trained Random Forest will guess your class.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 10, 100, 25)
        height = c2.number_input("Height (m)", 1.0, 2.5, 1.70, step=0.01)
        weight = c3.number_input("Weight (kg)", 30.0, 200.0, 70.0, step=0.5)

        c4, c5 = st.columns(2)
        gender = c4.selectbox("Gender", ["Female", "Male"])
        fam_hist = c5.selectbox("Family history of overweight?", ["yes", "no"])

        c6, c7 = st.columns(2)
        favc = c6.selectbox("Eat high-caloric food frequently?", ["yes", "no"])
        smoke = c7.selectbox("Do you smoke?", ["no", "yes"])

        c8, c9 = st.columns(2)
        fcvc = c8.slider("Vegetable consumption (FCVC)", 1.0, 3.0, 2.0, 0.1)
        ncp = c9.slider("Main meals per day (NCP)", 1.0, 4.0, 3.0, 0.1)

        c10, c11 = st.columns(2)
        ch2o = c10.slider("Water intake (CH2O)", 1.0, 3.0, 2.0, 0.1)
        faf = c11.slider("Physical activity freq (FAF)", 0.0, 3.0, 1.0, 0.1)

        c12, c13 = st.columns(2)
        tue = c12.slider("Technology use (TUE)", 0.0, 2.0, 1.0, 0.1)
        scc = c13.selectbox("Calorie monitoring? (SCC)", ["no", "yes"])

        c14, c15 = st.columns(2)
        caec = c14.selectbox("Snacking between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
        calc = c15.selectbox("Alcohol consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])

        mtrans = st.selectbox("Main transportation", [
            "Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"
        ])

        submitted = st.form_submit_button("Predict")

    if submitted:
        caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}

        row = {col: 0 for col in X_encoded.columns}
        row["Age"] = age
        row["Height"] = height
        row["Weight"] = weight
        row["FCVC"] = fcvc
        row["NCP"] = ncp
        row["CH2O"] = ch2o
        row["FAF"] = faf
        row["TUE"] = tue
        row["CAEC"] = caec_map[caec]
        row["CALC"] = calc_map[calc]

        gender_col = f"Gender_{gender}"
        if gender_col in row:
            row[gender_col] = 1
        fam_col = f"family_history_with_overweight_{fam_hist}"
        if fam_col in row:
            row[fam_col] = 1
        favc_col = f"FAVC_{favc}"
        if favc_col in row:
            row[favc_col] = 1
        smoke_col = f"SMOKE_{smoke}"
        if smoke_col in row:
            row[smoke_col] = 1
        scc_col = f"SCC_{scc}"
        if scc_col in row:
            row[scc_col] = 1
        mtrans_col = f"MTRANS_{mtrans}"
        if mtrans_col in row:
            row[mtrans_col] = 1

        input_df = pd.DataFrame([row])
        cont = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        input_df[cont] = scaler.transform(input_df[cont])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        predicted_class = class_labels[pred]

        st.success(f"**Predicted class: {pretty(predicted_class)}**")

        proba_df = pd.DataFrame({"Class": pretty_labels, "Probability": proba})
        fig = px.bar(
            proba_df, x="Class", y="Probability",
            color="Probability",
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_tickangle=-30,
            yaxis_range=[0, 1],
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
