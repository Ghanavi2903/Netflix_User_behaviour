"""
Netflix Customer Churn Prediction Dashboard.Predict user behaviour.
Run: streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure src is importable regardless of CWD
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import (
    preprocessing, eda, probability, simulation,
    models, statistical_tests, ab_testing, evaluation,
)

DATA_PATH = ROOT / "data" / "netflix_user_behavior_churn_50000v2.csv"

# ---------------------------------------------------------------- page config
st.set_page_config(
    page_title="Netflix Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title   {font-size:2.2rem; font-weight:700; color:#E50914;}
    .sub-title    {color:#888; margin-bottom:1rem;}
    div[data-testid="stMetricValue"] {font-size:1.5rem;}
    .section-title{font-size:1.4rem; font-weight:600; border-left:5px solid #E50914;
                   padding-left:10px; margin:1rem 0;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def load():
    if not DATA_PATH.exists():
        return None
    return preprocessing.load_data(str(DATA_PATH))


df_raw = load()

if df_raw is None:
    st.error(f"Dataset not found at:\n\n`{DATA_PATH}`")
    st.info("Place `netflix_user_behavior_churn_50000v2.csv` inside the `data/` folder and refresh.")
    st.stop()

df = df_raw.copy()


# ---------------------------------------------------------------- sidebar nav
st.sidebar.markdown("## Netflix Churn")
st.sidebar.caption(f"Dataset: **{df.shape[0]:,}** rows x **{df.shape[1]}** cols")
page = st.sidebar.radio(
    "Go to",
    [
        "Home & Dataset",
        "Exploratory Data Analysis",
        "Probability Analysis",
        "Random Variable Simulation",
        "Predict a Customer",
        "Classification Models",
        "Statistical Tests",
        "A/B Testing",
        "Model Evaluation",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info("Select a section above. All 8 required sections are implemented.")


# ================================================================ HOME
def render_home():
    st.markdown('<div class="main-title">Netflix Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An end-to-end ML, statistics and simulation dashboard.</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", f"{len(df):,}")
    c2.metric("Churn Rate", f"{df['churned'].mean()*100:.1f}%")
    c3.metric("Avg Weekly Watch", f"{df['avg_watch_time_minutes_per_week'].mean():.0f} min")
    c4.metric("Features", f"{df.shape[1]}")

    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.write("**Shape:**", df.shape)
    st.dataframe(df.head(20), use_container_width=True)
    st.dataframe(preprocessing.dataset_overview(df), use_container_width=True)

    st.markdown('<div class="section-title">Data Quality</div>', unsafe_allow_html=True)
    q = preprocessing.data_quality(df)
    col1, col2 = st.columns(2)
    col1.metric("Duplicate rows", q["duplicate_rows"])
    col2.metric("Duplicate user_ids", q["duplicate_user_ids"])
    if not q["missing"].empty:
        st.write("Missing values:")
        st.dataframe(q["missing"], use_container_width=True)
    else:
        st.success("No missing values.")

    st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(preprocessing.descriptive_stats(df), use_container_width=True)

    st.info("**Note:** The dataset does not contain an explicit `monthly_fee` column. "
            "A derived `monthly_fee` is created from `subscription_type` "
            "(Basic=$9.99, Standard=$15.49, Premium=$22.99) so the requested "
            "fee-related plots can be produced.")


# ================================================================ EDA
def render_eda():
    st.markdown('<div class="main-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">1. Churn Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(eda.churn_distribution(df), use_container_width=True)

    st.markdown('<div class="section-title">2. Age Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(eda.age_histogram(df), use_container_width=True)

    st.markdown('<div class="section-title">3. Monthly Fee Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(eda.monthly_fee_histogram(df), use_container_width=True)

    st.markdown('<div class="section-title">4. Watch Time Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(eda.watch_time_histogram(df), use_container_width=True)

    st.markdown('<div class="section-title">5. Monthly Fee vs Churn</div>', unsafe_allow_html=True)
    st.plotly_chart(eda.box_monthly_fee_vs_churn(df), use_container_width=True)

    st.markdown('<div class="section-title">6. Watch Time vs Churn</div>', unsafe_allow_html=True)
    st.plotly_chart(eda.box_watch_time_vs_churn(df), use_container_width=True)

    st.markdown('<div class="section-title">7. Subscription Type vs Churn</div>', unsafe_allow_html=True)
    st.plotly_chart(eda.subscription_vs_churn(df), use_container_width=True)

    st.markdown('<div class="section-title">8. Correlation Heatmap</div>', unsafe_allow_html=True)
    st.plotly_chart(eda.correlation_heatmap(df), use_container_width=True)

    st.markdown('<div class="section-title">Bivariate Analysis - Feature vs Churn</div>', unsafe_allow_html=True)
    cat_options = [c for c in preprocessing.CATEGORICAL_COLS if c in df.columns]
    pick = st.selectbox("Categorical feature", cat_options, index=0)
    t = pd.crosstab(df[pick], df["churned"], normalize="index") * 100
    t.columns = ["Not Churned %", "Churned %"]
    st.dataframe(t.round(2), use_container_width=True)

    st.markdown('<div class="section-title">Outlier Analysis (IQR)</div>', unsafe_allow_html=True)
    st.dataframe(eda.outlier_summary(df), use_container_width=True)
    st.plotly_chart(eda.outlier_boxplots(df), use_container_width=True)


# ================================================================ PROBABILITY
def render_probability():
    st.markdown('<div class="main-title">Probability and Sampling</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sample mean, variance, standard error, CI</div>',
                unsafe_allow_html=True)
    col = st.selectbox(
        "Column to sample",
        ["avg_watch_time_minutes_per_week", "age", "monthly_fee",
         "days_since_last_login", "account_age_months"],
        index=0,
    )
    n = st.slider("Sample size (n)", 30, 2000, 300, step=10)
    res = probability.sample_estimates(df, col, n, n_samples=500)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Population mean", f"{res['population_mean']:.2f}")
    c2.metric("Avg of sample means", f"{res['sample_mean_of_means']:.2f}")
    c3.metric("Sample variance", f"{res['sample_variance_of_means']:.4f}")
    c4.metric("Std error", f"{res['standard_error']:.4f}")

    ci = probability.confidence_interval(df[col].dropna().to_numpy(), 0.95)
    st.success(f"95% CI for mean of **{col}**: [{ci['lower']:.2f}, {ci['upper']:.2f}]  "
               f"(mean = {ci['mean']:.2f}, se = {ci['se']:.2f})")

    fig = px.histogram(res["means"], nbins=40,
                       title=f"Sampling distribution of the mean (n={n}, {res['n_samples']} draws)")
    fig.add_vline(x=res["population_mean"], line_dash="dash", line_color="red",
                  annotation_text="Population mean")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Conditional Probability P(Churn | Group)</div>',
                unsafe_allow_html=True)
    group_col = st.selectbox("Group column", ["subscription_type", "region", "payment_method",
                                              "primary_device", "gender", "favorite_genre"])
    values = df[group_col].dropna().unique().tolist()
    rows = [probability.conditional_probability(df, group_col, v) for v in values]
    cp = pd.DataFrame(rows).sort_values("p", ascending=False)
    st.dataframe(cp.style.format({"p": "{:.3f}"}), use_container_width=True)
    fig = px.bar(cp, x="group", y="p", text=cp["p"].round(3),
                 title=f"P(Churn | {group_col} = x)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Bayes Theorem</div>', unsafe_allow_html=True)
    target_val = st.selectbox("Group value", values, index=0)
    b = probability.bayes_theorem(df, group_col, target_val)
    st.latex(r"P(\text{churn} \mid g) = \frac{P(g \mid \text{churn}) \cdot P(\text{churn})}{P(g)}")
    st.json({
        "P(churn)": b["p_target"],
        f"P({target_val} | churn)": b["p_group_given_target"],
        f"P({target_val})": b["p_group"],
        f"P(churn | {target_val})": b["p_target_given_group"],
    })


# ================================================================ SIMULATION
def render_simulation():
    st.markdown('<div class="main-title">Random Variable Simulation</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Binomial - number of churners in a sample</div>',
                unsafe_allow_html=True)
    p_default = float(df["churned"].mean())
    c1, c2 = st.columns(2)
    n_trials = c1.slider("Users in each simulated group", 50, 2000, 500, step=10)
    p_success = c2.slider("Churn probability p", 0.01, 0.95, round(p_default, 3), step=0.01)
    n_samples = st.slider("Number of simulations", 200, 5000, 1000, step=100)

    sim = simulation.binomial_simulation(n_trials, p_success, n_samples)
    c1, c2, c3 = st.columns(3)
    c1.metric("Simulated mean", f"{sim['mean']:.1f}")
    c2.metric("Theoretical mean  n*p", f"{sim['theoretical_mean']:.1f}")
    c3.metric("Theoretical std  sqrt(np(1-p))", f"{sim['theoretical_std']:.2f}")

    fig = px.histogram(sim["samples"], nbins=40,
                       title=f"Binomial(n={n_trials}, p={p_success})")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Continuous distributions</div>', unsafe_allow_html=True)
    dist = st.radio("Distribution", ["Normal (watch time)", "Exponential (time gaps)"],
                    horizontal=True)
    col = df["avg_watch_time_minutes_per_week"].dropna()
    if dist.startswith("Normal"):
        samples = simulation.normal_simulation(col.mean(), col.std(), 2000)
        fig = px.histogram(samples, nbins=40,
                           title=f"Normal(mu={col.mean():.1f}, sigma={col.std():.1f})")
    else:
        samples = simulation.exponential_simulation(col.mean(), 2000)
        fig = px.histogram(samples, nbins=40, title=f"Exponential(scale={col.mean():.1f})")
    st.plotly_chart(fig, use_container_width=True)


# ================================================================ CLASSIFICATION
def get_model_bundle():
    if "bundle" not in st.session_state:
        X_tr, X_te, y_tr, y_te, pre, feat_names = preprocessing.encode_and_split(df)
        bundle = models.train_models(X_tr, X_te, y_tr, y_te)
        bundle.update({
            "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
            "preprocessor": pre, "feature_names": feat_names,
        })
        st.session_state["bundle"] = bundle
    return st.session_state["bundle"]


def render_classification():
    st.markdown('<div class="main-title">Classification Models</div>', unsafe_allow_html=True)

    bundle = get_model_bundle()
    y_test = bundle["y_test"]

    st.markdown('<div class="section-title">Metrics summary</div>', unsafe_allow_html=True)
    lr_eval = evaluation.evaluate(bundle["lr"], bundle["X_test"], y_test, "Logistic Regression")
    dt_eval = evaluation.evaluate(bundle["dt"], bundle["X_test"], y_test, "Decision Tree")

    rows = []
    for r in (lr_eval, dt_eval):
        rows.append({
            "Model": r["name"], "Accuracy": f"{r['accuracy']:.3f}",
            "Precision": f"{r['precision']:.3f}", "Recall/Sensitivity": f"{r['recall']:.3f}",
            "Specificity": f"{r['specificity']:.3f}", "F1": f"{r['f1']:.3f}",
            "ROC-AUC": f"{r['roc_auc']:.3f}", "Cross-entropy": f"{r['cross_entropy']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown('<div class="section-title">Entropy and Information Gain</div>',
                unsafe_allow_html=True)
    y = bundle["y_train"]
    base_entropy = models.entropy(y)
    st.write(f"**Base entropy of target (churned):** `{base_entropy:.4f}` bits")

    X_train = bundle["X_train"]
    feat_names = bundle["feature_names"]
    gains = []
    for i, name in enumerate(feat_names):
        gains.append({"feature": name,
                      "information_gain": models.information_gain(X_train[:, i], y)})
    gains_df = pd.DataFrame(gains).sort_values("information_gain", ascending=False).head(15)
    st.dataframe(gains_df.round(4), use_container_width=True)
    st.plotly_chart(px.bar(gains_df, x="information_gain", y="feature", orientation="h",
                           title="Top-15 features by information gain"),
                    use_container_width=True)

    st.markdown('<div class="section-title">Decision Tree - feature importance</div>',
                unsafe_allow_html=True)
    fi = pd.DataFrame({
        "feature": feat_names,
        "importance": bundle["dt"].feature_importances_,
    }).sort_values("importance", ascending=False).head(15)
    st.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h"),
                    use_container_width=True)


# ================================================================ STATS TESTS
def render_stats():
    st.markdown('<div class="main-title">Statistical Hypothesis Testing</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Two-sample t-test (watch time by churn)</div>',
                unsafe_allow_html=True)
    tt = statistical_tests.two_sample_t_test(df, "avg_watch_time_minutes_per_week")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean (not churned)", f"{tt['mean_non_churn']:.1f}")
    c2.metric("Mean (churned)", f"{tt['mean_churn']:.1f}")
    c3.metric("t-statistic", f"{tt['t_statistic']:.3f}")
    st.write(f"**p-value:** `{tt['p_value']:.2e}` - "
             f"{'significant' if tt['significant'] else 'not significant'}")
    st.info("H0: watch-time means are equal between churned and non-churned users.")

    st.markdown('<div class="section-title">Chi-square test (subscription vs churn)</div>',
                unsafe_allow_html=True)
    cs = statistical_tests.chi_square_test(df, "subscription_type")
    st.dataframe(cs["table"], use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Chi-square", f"{cs['chi2']:.3f}")
    c2.metric("dof", cs["dof"])
    c3.metric("Cramers V", f"{cs['cramers_v']:.3f}")
    st.write(f"**p-value:** `{cs['p_value']:.3e}` - "
             f"{'significant' if cs['significant'] else 'not significant'}")
    st.info("H0: subscription type and churn are independent.")


# ================================================================ A/B
def render_ab():
    st.markdown('<div class="main-title">A/B Testing - Discount Experiment</div>',
                unsafe_allow_html=True)
    st.write("Simulate whether a hypothetical discount reduces churn.")

    baseline = float(df["churned"].mean())
    c1, c2 = st.columns(2)
    effect = c1.slider("Absolute reduction in churn (discount effect)", 0.0, 0.20,
                       value=0.05, step=0.01)
    n_users = c2.slider("Total users in experiment", 500, len(df), min(5000, len(df)), step=100)

    res = ab_testing.run_experiment(baseline, effect, n_users)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control A churn", f"{res['rate_a']*100:.2f}%")
    c2.metric("Treatment B churn", f"{res['rate_b']*100:.2f}%")
    c3.metric("Absolute lift", f"{res['absolute_lift']*100:.2f} pp")
    c4.metric("p-value", f"{res['p_value']:.3e}")

    chart = pd.DataFrame({
        "group": ["A (control)", "B (discount)"],
        "churn_rate": [res["rate_a"] * 100, res["rate_b"] * 100],
    })
    st.plotly_chart(px.bar(chart, x="group", y="churn_rate", text="churn_rate",
                           title="Churn rate by group (%)"),
                    use_container_width=True)

    verdict = "The discount significantly reduces churn." if res["significant"] \
        else "No statistically significant difference (at alpha = 0.05)."
    st.success(verdict)


# ================================================================ MODEL EVAL
def render_evaluation():
    st.markdown('<div class="main-title">Model Evaluation</div>', unsafe_allow_html=True)
    bundle = get_model_bundle()
    y_test = bundle["y_test"]

    for key, name in [("lr", "Logistic Regression"), ("dt", "Decision Tree")]:
        st.markdown(f'<div class="section-title">{name}</div>', unsafe_allow_html=True)
        r = evaluation.evaluate(bundle[key], bundle["X_test"], y_test, name)

        cm = r["cm"]
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                             columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{r['accuracy']:.3f}")
        c2.metric("Precision", f"{r['precision']:.3f}")
        c3.metric("Sensitivity / Recall", f"{r['recall']:.3f}")
        c4.metric("Specificity", f"{r['specificity']:.3f}")
        c5.metric("F1", f"{r['f1']:.3f}")
        c6.metric("AUC", f"{r['roc_auc']:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=r["roc_fpr"], y=r["roc_tpr"], mode="lines",
                                         name=f"AUC={r['roc_auc']:.3f}"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash", color="gray"),
                                         name="Random"))
            fig_roc.update_layout(title=f"ROC curve - {name}",
                                  xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
        with col2:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=r["pr_recall"], y=r["pr_precision"],
                                        mode="lines", name="Precision-Recall"))
            fig_pr.update_layout(title=f"Precision-Recall curve - {name}",
                                 xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, use_container_width=True)

        st.info(f"Cross-entropy (log-loss) - {name}: **{r['cross_entropy']:.4f}**")


# ================================================================ PREDICT
def render_predict():
    st.markdown('<div class="main-title">Predict Customer Churn</div>', unsafe_allow_html=True)
    st.write("Enter a customer's details below to predict their churn probability using the trained models.")

    # Get the cached models
    bundle = get_model_bundle()

    # Create a UI Form
    with st.form("predict_form"):
        st.markdown("### Customer Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["Europe", "Asia", "North America", "South America", "Africa", "Oceania"])
            sub = st.selectbox("Subscription", ["Basic", "Standard", "Premium"])
            payment = st.selectbox("Payment Method", ["Credit Card", "Paypal", "Debit Card", "Gift Card"])
            device = st.selectbox("Primary Device", ["Mobile", "Smart TV", "Laptop", "Tablet"])

        with c2:
            account_age = st.slider("Account Age (months)", 0, 130, 30)
            genre = st.selectbox("Favorite Genre", ["Comedy", "Drama", "Action", "Romance", "Horror", "Sci-Fi", "Documentary"])
            time_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
            rec_src = st.selectbox("Recommendation Source", ["Algorithm", "Search", "Trending", "Friend", "Ads"])
            sessions = st.slider("Session Count", 1, 5, 1)
            watch_sess = st.slider("Watch Sessions / Week", 1, 19, 4)

        with c3:
            watch = st.slider("Avg Watch Time (min/week)", 0, 820, 200)
            completion = st.slider("Completion Rate (%)", 30, 100, 75)
            avg_rating = st.slider("Avg Rating Given (1-5)", 1, 5, 4)
            app_rating = st.slider("App Rating (1-5)", 1, 5, 4)
            click_rate = st.slider("Recommendation Click Rate (%)", 0, 100, 40)
            days_inactive = st.slider("Days Since Last Login", 0, 60, 5)

        # Submit Button
        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    # Logic after user clicks Predict
    if submitted:
        # 1. Build a dataframe for the single user
        customer = pd.DataFrame([{
            "age": age, "gender": gender, "region": region,
            "subscription_type": sub, "payment_method": payment,
            "primary_device": device, "account_age_months": account_age,
            "favorite_genre": genre, "time_of_day": time_day,
            "recommendation_source": rec_src, "session_count": sessions,
            "avg_watch_time_minutes_per_week": watch,
            "watch_sessions_per_week": watch_sess,
            "completion_rate": completion, "avg_rating_given": avg_rating,
            "app_rating": app_rating,
            "recommendation_click_rate": click_rate,
            "days_since_last_login": days_inactive,
        }])

        # 2. Add derived monthly_fee
        customer["monthly_fee"] = customer["subscription_type"].map(preprocessing.SUBSCRIPTION_FEE)

        # 3. Encode the input using the trained preprocessor
        X_new = bundle["preprocessor"].transform(customer)

        # 4. Predict probabilities
        prob_lr = float(bundle["lr"].predict_proba(X_new)[0, 1])
        prob_dt = float(bundle["dt"].predict_proba(X_new)[0, 1])
        avg_prob = (prob_lr + prob_dt) / 2

        # 5. Display Results
        st.markdown("---")
        st.markdown("### Prediction Results")

        # Verdict Banner
        if avg_prob >= 0.5:
            st.error(f"**HIGH CHURN RISK** - {avg_prob*100:.1f}% probability")
            st.write("**Action Required:** Offer a targeted discount or personalized content recommendation.")
        elif avg_prob >= 0.3:
            st.warning(f"**MODERATE RISK** - {avg_prob*100:.1f}% probability")
            st.write("**Action Required:** Monitor this customer's engagement closely.")
        else:
            st.success(f"**LOW RISK** - {avg_prob*100:.1f}% probability")
            st.write("**Action Required:** Customer is likely to stay. No immediate action needed.")

        # Two columns: Probabilities & Chart
        colA, colB = st.columns(2)

        with colA:
            st.metric("Logistic Regression", f"{prob_lr*100:.1f}%")
            st.metric("Decision Tree", f"{prob_dt*100:.1f}%")

        with colB:
            fig = go.Figure(go.Bar(
                x=[prob_lr*100, prob_dt*100],
                y=["Logistic Reg.", "Decision Tree"],
                orientation="h",
                marker_color=["#E50914" if p >= 0.5 else "#2ecc71" for p in [prob_lr, prob_dt]],
                text=[f"{prob_lr*100:.1f}%", f"{prob_dt*100:.1f}%"],
                textposition="outside",
            ))
            fig.update_layout(title="Model Probabilities", xaxis=dict(range=[0, 100]), height=200, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Why? (Comparison to dataset medians)
        st.markdown("#### Why did the model predict this?")
        st.caption("Comparing this customer's key features to the dataset median:")

        baseline = {
            "avg_watch_time_minutes_per_week": df["avg_watch_time_minutes_per_week"].median(),
            "days_since_last_login": df["days_since_last_login"].median(),
            "completion_rate": df["completion_rate"].median(),
            "monthly_fee": df["monthly_fee"].median(),
        }
        customer_vals = {
            "avg_watch_time_minutes_per_week": watch,
            "days_since_last_login": days_inactive,
            "completion_rate": completion,
            "monthly_fee": customer["monthly_fee"].iloc[0],
        }

        diff = pd.DataFrame([
            {
                "Feature": k.replace("_", " ").title(),
                "Customer Value": round(customer_vals[k], 1),
                "Dataset Median": round(baseline[k], 1),
                "Difference": round(customer_vals[k] - baseline[k], 1)
            }
            for k in baseline
        ])
        st.dataframe(diff, use_container_width=True, hide_index=True)


# ================================================================ router
if   page == "Home & Dataset":                render_home()
elif page == "Exploratory Data Analysis":     render_eda()
elif page == "Probability Analysis":          render_probability()
elif page == "Random Variable Simulation":    render_simulation()
elif page == "Predict a Customer":            render_predict()
elif page == "Classification Models":         render_classification()
elif page == "Statistical Tests":             render_stats()
elif page == "A/B Testing":                   render_ab()
elif page == "Model Evaluation":              render_evaluation()