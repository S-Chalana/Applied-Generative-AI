"""
Banking Customer Support AI - Streamlit Application
Interactive web interface for the customer support agent system.
"""

import streamlit as st
import pandas as pd
import re
from main_agents import (
    State,
    build_workflow,
    save_to_history,
    get_agent_metrics,
    evaluate_response,
    evaluate_routing_accuracy,
    get_few_shot_examples,
    TEST_SCENARIOS
)

def run_streamlit_app():
    """Main Streamlit application interface."""
    # ── Global styles ────────────────────────────────────────────────────
    st.markdown("""
        <style>
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e8eaf0;
            border-radius: 12px;
            padding: 20px 24px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 8px;
        }
        .kpi-label {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 36px;
            font-weight: 800;
            color: #111827;
            line-height: 1.1;
        }
        .kpi-sub {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 4px;
        }
        .section-label {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            color: #6366f1;
            margin-bottom: 4px;
        }
        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 16px;
        }
        .divider { border-top: 1px solid #e8eaf0; margin: 24px 0; }
        .response-box {
            background: #f0f4ff;
            border-left: 4px solid #6366f1;
            border-radius: 8px;
            padding: 24px 28px;
            font-size: 20px;
            line-height: 1.7;
            color: #1e293b;
            margin-bottom: 12px;
        }
        .hero-banner {
            background: linear-gradient(135deg, #0d0d1a 0%, #1a1040 60%, #0f1e3d 100%);
            border-radius: 16px;
            padding: 48px 40px 40px 40px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .hero-eyebrow {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #a78bfa;
            margin-bottom: 12px;
        }
        .hero-title {
            font-size: 42px;
            font-weight: 900;
            color: #ffffff;
            line-height: 1.15;
            margin-bottom: 14px;
            letter-spacing: -0.5px;
        }
        .hero-title span {
            background: linear-gradient(90deg, #818cf8, #a78bfa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub {
            font-size: 16px;
            color: #94a3b8;
            line-height: 1.6;
            max-width: 560px;
        }
        /* hide default Streamlit title */
        h1[data-testid="stHeading"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)

    # Hero banner (replaces st.title)
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-eyebrow">AI-POWERED · BANKING · CUSTOMER SUPPORT</div>
            <div class="hero-title">Banking Customer<br><span>Support AI</span></div>
            <div class="hero-sub">Intelligent routing, real-time ticket lookup, and continuous learning — all in one platform.</div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    with st.sidebar:
        page = st.radio("Go to:", ["Home", "Model Evaluation", "History & Logs"])
    
    # Page 1: Chat Interface
    if page == "Home":
        user_input = st.text_area("✏️ Enter customer message:", height=120, placeholder="e.g. I love the service! / My card was charged twice. / What's the status of ticket T001?")

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            submitted = st.button("🚀 Submit", type="primary", use_container_width=True)

        if submitted:
            if user_input:
                with st.spinner("Routing to agent..."):
                    initial_state: State = {"user_input": user_input, "decision": "", "output": "", "evaluation": {}}
                    history = st.session_state.get('history', [])
                    workflow = build_workflow(history=history)
                    final_state = workflow.invoke(initial_state)

                    ticket_id = re.search(r'T\d+', user_input)
                    trace_data = {
                        "classification_prompt": f"Classify: '{user_input}'",
                        "classification_output": final_state['decision'],
                        "agent_handler": f"handle_{final_state['decision']}",
                        "ticket_action": f"Queried tickets.csv for {ticket_id.group(0)}" if ticket_id else "No database query"
                    }
                    save_to_history(final_state, trace_data=trace_data, session_state=st.session_state)
                    st.session_state['last_interaction'] = final_state
                    st.session_state['last_trace'] = trace_data

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-label">AGENT RESPONSE</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="response-box">{final_state["output"]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-label">FEEDBACK</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Rate this response</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful", use_container_width=True):
                        if 'history' in st.session_state and st.session_state['history']:
                            st.session_state['history'][-1]['feedback'] = 'positive'
                            st.session_state['history'][-1]['success'] = True
                            st.success("Thanks for your feedback!")
                with col2:
                    if st.button("👎 Not Helpful", use_container_width=True):
                        if 'history' in st.session_state and st.session_state['history']:
                            st.session_state['history'][-1]['feedback'] = 'negative'
                            st.session_state['history'][-1]['success'] = False
                            st.warning("Feedback recorded for improvement")

                user_feedback_text = st.text_area("Additional comments (optional):", placeholder="Tell us what you liked or what could be improved...")
                if st.button("Submit Comments") and user_feedback_text:
                    if 'history' in st.session_state and st.session_state['history']:
                        st.session_state['history'][-1]['feedback_comments'] = user_feedback_text
                        st.success("Thank you for your detailed feedback!")
            else:
                st.warning("Please enter a message")
    
    # Page 2: Test Scenarios
    elif page == "Test Scenarios":
        st.subheader("🧪 Test Agent Scenarios")
        agent_type = st.selectbox("Select Agent Type:", ["positive", "negative", "query"])
        
        st.write(f"**Pre-built test cases for {agent_type} agent:**")
        for scenario in TEST_SCENARIOS[agent_type]:
            if st.button(f"Test: {scenario}", key=scenario):
                initial_state: State = {"user_input": scenario, "decision": "", "output": "", "evaluation": {}}
                # Pass history to workflow for few-shot learning
                history = st.session_state.get('history', [])
                workflow = build_workflow(history=history)
                final_state = workflow.invoke(initial_state)
                save_to_history(final_state, session_state=st.session_state)
                
                col1, col2 = st.columns(2)
                col1.metric("Expected", agent_type)
                col2.metric("Actual", final_state['decision'])
                st.write(f"**Response:** {final_state['output']}")
                if final_state['decision'] == agent_type:
                    st.success("✅ Routing Correct")
                else:
                    st.error("❌ Routing Failed")
    
    # Page 3: History & Logs
    elif page == "History & Logs":
        if 'history' in st.session_state and st.session_state['history']:
            df = pd.DataFrame(st.session_state['history'])
            metrics = get_agent_metrics(session_state=st.session_state)

            # ── KPI BANNER ──────────────────────────────────────────────────
            st.markdown('<div class="section-label">PERFORMANCE OVERVIEW</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Agent Analytics Dashboard</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Total Interactions</div>
                    <div class="kpi-value">{metrics['total_interactions']}</div>
                    <div class="kpi-sub">All time</div></div>""", unsafe_allow_html=True)
            with c2:
                rate = metrics['success_rate']
                color = "#16a34a" if rate >= 70 else "#d97706" if rate >= 40 else "#dc2626"
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Success Rate</div>
                    <div class="kpi-value" style="color:{color}">{rate:.1f}%</div>
                    <div class="kpi-sub">Based on feedback</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Tickets Processed</div>
                    <div class="kpi-value">{metrics['tickets_processed']}</div>
                    <div class="kpi-sub">DB queries made</div></div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Unique Tickets</div>
                    <div class="kpi-value">{metrics['unique_tickets']}</div>
                    <div class="kpi-sub">Distinct IDs</div></div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # ── AGENT DISTRIBUTION ───────────────────────────────────────────
            st.markdown('<div class="section-label">ROUTING BREAKDOWN</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Agent Distribution</div>', unsafe_allow_html=True)
            counts = df['decision'].value_counts()
            total = counts.sum()
            cols = st.columns(3)
            tags = {"positive": ("#d1fae5", "#065f46", "👍 Positive"),
                    "negative": ("#fee2e2", "#991b1b", "👎 Negative"),
                    "query":    ("#dbeafe", "#1e40af", "🔍 Query")}
            for i, (agent, (bg, fg, label)) in enumerate(tags.items()):
                count = counts.get(agent, 0)
                pct = count / total * 100 if total else 0
                with cols[i]:
                    st.markdown(f"""<div class="kpi-card">
                        <div class="kpi-label" style="color:{fg}">{label}</div>
                        <div class="kpi-value" style="color:{fg}">{count}</div>
                        <div class="kpi-sub">{pct:.1f}% of total</div></div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # ── IMPROVEMENT LOOP ─────────────────────────────────────────────
            st.markdown('<div class="section-label">FEW-SHOT LEARNING</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Agent Improvement Loop</div>', unsafe_allow_html=True)
            st.info("The system learns from your feedback. Interactions marked 👍 are used as in-context examples to improve future responses.")

            feedback_df = df[df['feedback'].notna()]
            if len(feedback_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Feedback by Agent Type**")
                    agent_feedback = feedback_df.groupby(['decision', 'feedback']).size().unstack(fill_value=0)
                    st.dataframe(agent_feedback, use_container_width=True)
                with col2:
                    st.markdown("**Learning Examples Active**")
                    for agent_type in ['positive', 'negative', 'query']:
                        examples = get_few_shot_examples(agent_type, history=st.session_state.get('history', []))
                        if examples:
                            st.success(f"✅ **{agent_type.capitalize()}** — {len(examples)} example(s) in use")
                        else:
                            st.warning(f"⏳ **{agent_type.capitalize()}** — no examples yet")

                st.markdown("**Success Rate Trend**")
                feedback_df = feedback_df.copy()
                feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
                daily_success = feedback_df.groupby('date')['feedback'].apply(
                    lambda x: (x == 'positive').sum() / len(x) * 100 if len(x) > 0 else 0
                )
                if len(daily_success) > 0:
                    st.line_chart(daily_success)
            else:
                st.warning("No feedback yet. Start providing feedback to enable the improvement loop!")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # ── TICKET LOG ───────────────────────────────────────────────────
            if metrics['tickets_processed'] > 0:
                st.markdown('<div class="section-label">TICKET ACTIVITY</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Ticket Log</div>', unsafe_allow_html=True)
                ticket_df = df[df['ticket_id'].notna()][['timestamp', 'ticket_id', 'decision', 'feedback']]
                st.dataframe(ticket_df, use_container_width=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # ── INTERACTION LOGS ─────────────────────────────────────────────
            st.markdown('<div class="section-label">INTERACTION HISTORY</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Interaction Logs</div>', unsafe_allow_html=True)

            col1, col2 = st.columns([3, 1])
            with col1:
                filter_decision = st.multiselect("Filter by Agent:", ["positive", "negative", "query"], default=["positive", "negative", "query"])
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state['history'] = []
                    st.rerun()

            filtered_df = df[df['decision'].isin(filter_decision)]
            st.dataframe(filtered_df[['timestamp', 'input', 'decision', 'ticket_id', 'feedback']], use_container_width=True)

            # ── TRACE DETAILS ────────────────────────────────────────────────
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">TRACE INSPECTOR</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prompt Traces & Classification Details</div>', unsafe_allow_html=True)

            if len(filtered_df) > 0:
                selected_idx = st.selectbox("Choose interaction:",
                                           range(len(filtered_df)),
                                           format_func=lambda i: f"{filtered_df.iloc[i]['timestamp']} — {filtered_df.iloc[i]['input'][:60]}...")
                if selected_idx is not None:
                    selected_row = filtered_df.iloc[selected_idx]
                    if selected_row.get('trace'):
                        st.json(selected_row['trace'])
                    else:
                        st.info("No trace data available for this interaction")
                    if selected_row.get('feedback_comments'):
                        st.markdown("**User Feedback Comments**")
                        st.write(selected_row['feedback_comments'])

            # ── DOWNLOAD + CHARTS ────────────────────────────────────────────
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-label">EXPORT</div>', unsafe_allow_html=True)
                csv = filtered_df.to_csv(index=False)
                st.download_button("⬇️ Download Logs as CSV", csv, "interaction_logs.csv", "text/csv", use_container_width=True)
            with col2:
                st.markdown('<div class="section-label">RECENT ACTIVITY</div>', unsafe_allow_html=True)
                activity = df.copy()
                activity['time'] = pd.to_datetime(activity['timestamp']).dt.strftime('%H:%M')
                activity_counts = activity.groupby('time').size()
                if len(activity_counts) > 1:
                    st.line_chart(activity_counts)
                else:
                    st.bar_chart(activity_counts)

        else:
            st.markdown("""
                <div style="text-align:center; padding: 60px 20px; color: #6b7280;">
                    <div style="font-size:48px;">💬</div>
                    <div style="font-size:18px; font-weight:600; margin-top:12px;">No interaction history yet</div>
                    <div style="font-size:14px; margin-top:6px;">Head to the Home page and submit a message to get started.</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Page 2: Model Evaluation
    elif page == "Model Evaluation":
        st.markdown('<div class="section-label">AI QUALITY ASSURANCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Evaluation Dashboard</div>', unsafe_allow_html=True)
        st.markdown("Assess response quality and verify routing accuracy against built-in QA test cases.")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if 'last_interaction' not in st.session_state:
            st.markdown("""
                <div style="text-align:center; padding: 60px 20px; color: #6b7280;">
                    <div style="font-size:48px;">📊</div>
                    <div style="font-size:18px; font-weight:600; margin-top:12px;">No interaction to evaluate</div>
                    <div style="font-size:14px; margin-top:6px;">Submit a message on the Home page first.</div>
                </div>
            """, unsafe_allow_html=True)
            return

        eval_tab1, eval_tab2 = st.tabs(["📝 Response Quality", "🎯 Routing Accuracy"])

        with eval_tab1:
            st.markdown('<div class="section-label">RESPONSE QUALITY</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Last Response Evaluation</div>', unsafe_allow_html=True)
            with st.spinner("Evaluating response..."):
                eval_data = evaluate_response(st.session_state['last_interaction'])['evaluation']

            c1, c2, c3 = st.columns(3)
            for col, key, label in [(c1, 'accuracy', 'Accuracy'), (c2, 'empathy', 'Empathy'), (c3, 'clarity', 'Clarity')]:
                val = eval_data.get(key, 0)
                color = "#16a34a" if val >= 4 else "#d97706" if val >= 3 else "#dc2626"
                with col:
                    st.markdown(f"""<div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value" style="color:{color}">{val}<span style="font-size:18px;color:#9ca3af">/5</span></div>
                        <div class="kpi-sub">{['', 'Poor', 'Fair', 'Good', 'Great', 'Excellent'][val] if 1 <= val <= 5 else '—'}</div>
                    </div>""", unsafe_allow_html=True)

            if eval_data.get('feedback'):
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-label">AI FEEDBACK</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="response-box">{eval_data["feedback"]}</div>', unsafe_allow_html=True)

        with eval_tab2:
            st.markdown('<div class="section-label">ROUTING ACCURACY</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">QA Test Results</div>', unsafe_allow_html=True)
            with st.spinner("Running 10 QA test cases..."):
                results = evaluate_routing_accuracy()

            rate = results['success_rate']
            color = "#16a34a" if rate >= 70 else "#d97706" if rate >= 40 else "#dc2626"
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Success Rate</div>
                    <div class="kpi-value" style="color:{color}">{rate:.1f}%</div>
                    <div class="kpi-sub">Overall accuracy</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Correct</div>
                    <div class="kpi-value" style="color:#16a34a">{results['correct']}</div>
                    <div class="kpi-sub">Passed</div></div>""", unsafe_allow_html=True)
            with c3:
                wrong = results['total'] - results['correct']
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Incorrect</div>
                    <div class="kpi-value" style="color:#dc2626">{wrong}</div>
                    <div class="kpi-sub">Failed</div></div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">TEST CASE BREAKDOWN</div>', unsafe_allow_html=True)
            df = pd.DataFrame(results['results'])
            st.dataframe(df.style.apply(lambda row: ['background-color: #d1fae5'] * len(row) if row['correct']
                                        else ['background-color: #fee2e2'] * len(row), axis=1), use_container_width=True)
            st.markdown('<div class="section-label">CLASSIFICATION DISTRIBUTION</div>', unsafe_allow_html=True)
            st.bar_chart(df['actual'].value_counts())

# Run the app - Streamlit executes this directly
run_streamlit_app()
