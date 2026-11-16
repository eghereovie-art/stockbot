"""
Web app for trading_agent.py (paper trading only), built with Streamlit.

Files needed in the same folder:  app.py, trading_agent.py, requirements.txt
Run locally:   streamlit run app.py
Password:      set APP_PASSWORD (Streamlit secrets or environment variable)
Optional:      TELEGRAM_TOKEN and TELEGRAM_CHAT_ID secrets give trade alerts
"""
import contextlib
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import trading_agent as ta

st.set_page_config(page_title="Paper Trading Agent", page_icon="📈", layout="centered")
cfg = ta.Config()
DEFAULT = "AAPL MSFT NVDA AMZN TSLA GOOGL META"


# ------------------------------------------------------------ password gate
def get_password():
    try:
        if "APP_PASSWORD" in st.secrets:
            return str(st.secrets["APP_PASSWORD"])
    except Exception:
        pass
    return os.environ.get("APP_PASSWORD")


def gate():
    password = get_password()
    if not password:
        st.error("No password set. Add APP_PASSWORD to the app's secrets, then reload.")
        st.stop()
    if st.session_state.get("ok"):
        return
    st.title("Paper Trading Agent")
    entered = st.text_input("Password", type="password")
    if entered and entered == password:
        st.session_state["ok"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()


gate()


# ------------------------------------------------------------------ helpers
def capture(fn, *args):
    """Run a function and return everything it printed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(*args)
        except Exception as e:
            print(f"Error: {e}")
    return buf.getvalue()


def parse_symbols(text):
    return list(dict.fromkeys(s.strip().upper() for s in text.replace(",", " ").split() if s.strip()))


@st.cache_data(ttl=900, show_spinner=False)
def get_signals(symbols: tuple):
    rows = []
    for sym in symbols:
        try:
            feats = ta.make_features(ta.load_prices(sym))
            valid = feats.dropna(subset=ta.FEATURES)
            model = ta.fit_model(valid.iloc[:-1])
            prob = float(model.predict_proba(valid.iloc[[-1]])[0, 1])
            zone = ("Buy zone" if prob > cfg.buy_threshold
                    else "Sell zone" if prob < cfg.sell_threshold else "Hold")
            rows.append({"Symbol": sym, "Price": round(float(valid["close"].iloc[-1]), 2),
                         "P(up tomorrow)": round(prob, 3), "Signal": zone})
        except Exception as e:
            rows.append({"Symbol": sym, "Price": None, "P(up tomorrow)": None, "Signal": f"Failed: {e}"})
    df = pd.DataFrame(rows)
    return df.sort_values("P(up tomorrow)", ascending=False, na_position="last").reset_index(drop=True)


def latest_prices(symbols):
    out = {}
    for sym in symbols:
        try:
            out[sym] = float(ta.load_prices(sym, years=1)["Close"].iloc[-1])
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------- UI
st.title("📈 Paper Trading Agent")
st.caption("Fake money only. Backtests showed no reliable edge over buy and hold, so treat this as a learning tool.")

symbols_text = st.text_input("Stocks to watch (separated by spaces)", DEFAULT)
symbols = parse_symbols(symbols_text)

tab_signals, tab_trade, tab_account, tab_backtest = st.tabs(["Signals", "Trade", "Account", "Backtest"])

with tab_signals:
    st.write("Ranks stocks by their estimated chance of rising tomorrow. Nothing is bought or sold here.")
    if st.button("Get signals", type="primary", disabled=not symbols):
        with st.spinner("Checking the market..."):
            st.session_state["signals"] = get_signals(tuple(symbols))
    if "signals" in st.session_state:
        st.dataframe(st.session_state["signals"], use_container_width=True, hide_index=True)

with tab_trade:
    st.write(
        f"The agent buys when the chance of rising is above {cfg.buy_threshold:.0%} and sells below "
        f"{cfg.sell_threshold:.0%} or after a {cfg.stop_loss_pct:.0%} loss. "
        f"It holds at most {cfg.max_positions} stocks, up to {cfg.max_position_pct:.0%} of the account each."
    )
    sure = st.checkbox("I understand this changes my fake account")
    if st.button("Run agent now", type="primary", disabled=not (symbols and sure)):
        with st.spinner("Running the agent..."):
            st.session_state["run_output"] = capture(ta.run_agent, symbols, cfg)
        st.cache_data.clear()
    if "run_output" in st.session_state:
        st.code(st.session_state["run_output"], language=None)

with tab_account:
    broker = ta.PaperBroker(cfg)
    held = list(broker.state["positions"])
    prices = latest_prices(held) if held else {}
    equity = broker.equity(prices)
    c1, c2 = st.columns(2)
    c1.metric("Total equity", f"${equity:,.2f}", f"{equity - cfg.start_cash:+,.2f} since start")
    c2.metric("Cash", f"${broker.state['cash']:,.2f}")

    st.subheader("Positions")
    if held:
        rows = []
        for sym, p in broker.state["positions"].items():
            now = prices.get(sym, p["avg_price"])
            rows.append({"Symbol": sym, "Shares": p["qty"], "Avg price": round(p["avg_price"], 2),
                         "Now": round(now, 2), "P/L %": round((now / p["avg_price"] - 1) * 100, 2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("No open positions.")

    st.subheader("Recent trades")
    log = broker.state["log"][-15:][::-1]
    if log:
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
    else:
        st.write("No trades yet.")

    with st.expander("Reset account"):
        st.write(f"This deletes all positions and trades and restores ${cfg.start_cash:,.0f} in fake cash.")
        confirm = st.checkbox("Yes, reset my fake account")
        if st.button("Reset", disabled=not confirm):
            path = Path(cfg.state_file)
            if path.exists():
                path.unlink()
            st.rerun()

with tab_backtest:
    st.write("Tests the strategy on past prices and compares it with buying and holding.")
    bt_text = st.text_input("Stocks to test (up to 5)", "AAPL MSFT NVDA")
    bt_symbols = parse_symbols(bt_text)[:5]
    if st.button("Run backtest", type="primary", disabled=not bt_symbols):
        for sym in bt_symbols:
            with st.spinner(f"Testing {sym}..."):
                st.code(capture(ta.backtest, sym, cfg), language=None)

st.divider()
st.caption("Educational tool. Not financial advice.")
