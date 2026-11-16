"""
Stock trading agent (PAPER TRADING ONLY by default).

What it does
  1. Downloads daily prices (yfinance)
  2. Builds features (momentum, trend, RSI, volatility)
  3. Trains a model to estimate P(price up tomorrow)
  4. Backtests the strategy (walk-forward, with trading costs)
  5. Runs against a simulated account: buys, sells, stop-losses, position limits

Usage
  pip install yfinance pandas numpy scikit-learn
  python trading_agent.py backtest AAPL
  python trading_agent.py run AAPL MSFT NVDA AMZN TSLA GOOGL META
      # ranks all stocks, holds the best few; run once a day after market close
  Alerts (optional): set TELEGRAM_TOKEN + TELEGRAM_CHAT_ID, and/or
  EMAIL_USER + EMAIL_APP_PASSWORD, as environment variables.
  python trading_agent.py status

IMPORTANT: no model can reliably predict markets. Backtests overstate real
performance. Paper trade for months before ever risking real money.
"""
import json
import os
import smtplib
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


# ----------------------------------------------------------------- config
@dataclass
class Config:
    buy_threshold: float = 0.55      # buy when P(up) above this
    sell_threshold: float = 0.48     # sell when P(up) below this
    stop_loss_pct: float = 0.07      # sell if down 7% from entry
    max_position_pct: float = 0.20   # max 20% of equity per stock
    max_positions: int = 3           # hold at most this many stocks at once
    cost_bps: float = 5.0            # trading cost per side, in basis points
    train_frac: float = 0.5          # backtest: initial training share
    retrain_every: int = 60          # backtest: retrain every N days
    start_cash: float = 10_000.0
    state_file: str = "paper_account.json"


FEATURES = ["ret1", "ret5", "ret20", "dist_sma20", "sma_cross", "rsi", "vol20"]


# ----------------------------------------------------------------- alerts
def notify(message: str):
    """Send an alert to Telegram and/or email, if configured via env vars.

    Telegram: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
    Email (Gmail): EMAIL_USER, EMAIL_APP_PASSWORD, optional EMAIL_TO
    """
    token = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            data = urllib.parse.urlencode({"chat_id": chat, "text": message}).encode()
            urllib.request.urlopen(
                f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=10
            )
        except Exception as e:
            print(f"(Telegram alert failed: {e})")

    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_APP_PASSWORD")
    if user and password:
        try:
            msg = EmailMessage()
            msg["Subject"] = "Trading agent alert"
            msg["From"] = user
            msg["To"] = os.environ.get("EMAIL_TO", user)
            msg.set_content(message)
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
                server.login(user, password)
                server.send_message(msg)
        except Exception as e:
            print(f"(Email alert failed: {e})")


# ------------------------------------------------------------------- data
def load_prices(symbol: str, years: int = 8) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=f"{years}y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["close"] = c
    f["ret1"] = c.pct_change()
    f["ret5"] = c.pct_change(5)
    f["ret20"] = c.pct_change(20)
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    f["dist_sma20"] = c / sma20 - 1
    f["sma_cross"] = sma20 / sma50 - 1
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f["vol20"] = f["ret1"].rolling(20).std()
    f["target"] = (c.shift(-1) > c).astype(float).where(c.shift(-1).notna())
    return f


# ------------------------------------------------------------------ model
class HeuristicModel:
    """Fallback if scikit-learn is missing: simple trend + RSI rule."""
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        trend = np.tanh(X["sma_cross"].values * 20)
        mean_rev = np.where(X["rsi"].values > 75, -0.1, np.where(X["rsi"].values < 25, 0.1, 0))
        p = np.clip(0.5 + 0.1 * trend + mean_rev, 0.01, 0.99)
        return np.column_stack([1 - p, p])


def fit_model(train: pd.DataFrame):
    train = train.dropna(subset=FEATURES + ["target"])
    if HAVE_SKLEARN:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
        model.fit(train[FEATURES].values, train["target"].values)
        return _SkWrap(model)
    return HeuristicModel().fit(train[FEATURES], train["target"])


class _SkWrap:
    def __init__(self, m):
        self.m = m

    def predict_proba(self, X):
        return self.m.predict_proba(X[FEATURES].values)


# --------------------------------------------------------------- backtest
def backtest(symbol: str, cfg: Config):
    feats = make_features(load_prices(symbol))
    valid = feats.dropna(subset=FEATURES).copy()
    n = len(valid)
    start = int(n * cfg.train_frac)

    probs = pd.Series(np.nan, index=valid.index)
    for i in range(start, n, cfg.retrain_every):
        model = fit_model(valid.iloc[: i - 1])  # exclude last row: its target peeks into test
        block = valid.iloc[i : i + cfg.retrain_every]
        probs.loc[block.index] = model.predict_proba(block)[:, 1]

    test = valid.iloc[start:].copy()
    test["prob"] = probs.iloc[start:]

    # position held from today's close to tomorrow's close, with hysteresis
    pos, positions = 0, []
    for p in test["prob"]:
        if pos == 0 and p > cfg.buy_threshold:
            pos = 1
        elif pos == 1 and p < cfg.sell_threshold:
            pos = 0
        positions.append(pos)
    test["pos"] = positions

    next_ret = test["close"].pct_change().shift(-1).fillna(0)
    trades = test["pos"].diff().abs().fillna(0)
    strat = test["pos"] * next_ret - trades * cfg.cost_bps / 10_000
    equity = (1 + strat).cumprod()
    hold = (1 + next_ret).cumprod()

    def stats(curve, rets):
        years = len(curve) / 252
        cagr = curve.iloc[-1] ** (1 / years) - 1
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        dd = (curve / curve.cummax() - 1).min()
        return cagr, sharpe, dd

    s, h = stats(equity, strat), stats(hold, next_ret)
    print(f"\n{symbol} backtest ({test.index[0].date()} -> {test.index[-1].date()})")
    print(f"{'':14}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}")
    print(f"{'Strategy':14}{s[0]:>9.1%}{s[1]:>9.2f}{s[2]:>9.1%}")
    print(f"{'Buy & hold':14}{h[0]:>9.1%}{h[1]:>9.2f}{h[2]:>9.1%}")
    print(f"Trades: {int(trades.sum())}  |  model: {'logistic regression' if HAVE_SKLEARN else 'heuristic'}")
    if s[1] <= h[1]:
        print("Note: strategy did not beat buy & hold on risk-adjusted terms.")


# ----------------------------------------------------------------- broker
class PaperBroker:
    """Simulated account saved to a JSON file."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = Path(cfg.state_file)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"cash": cfg.start_cash, "positions": {}, "log": []}

    def save(self):
        self.path.write_text(json.dumps(self.state, indent=2))

    def equity(self, prices: dict) -> float:
        total = self.state["cash"]
        for sym, p in self.state["positions"].items():
            total += p["qty"] * prices.get(sym, p["avg_price"])
        return total

    def buy(self, sym, price, qty):
        cost = qty * price * (1 + self.cfg.cost_bps / 10_000)
        if qty <= 0 or cost > self.state["cash"]:
            return False
        self.state["cash"] -= cost
        pos = self.state["positions"].get(sym, {"qty": 0, "avg_price": 0.0})
        total_qty = pos["qty"] + qty
        pos["avg_price"] = (pos["qty"] * pos["avg_price"] + qty * price) / total_qty
        pos["qty"] = total_qty
        self.state["positions"][sym] = pos
        self._log("BUY", sym, qty, price)
        return True

    def sell_all(self, sym, price, reason):
        pos = self.state["positions"].pop(sym, None)
        if not pos:
            return False
        self.state["cash"] += pos["qty"] * price * (1 - self.cfg.cost_bps / 10_000)
        self._log(f"SELL ({reason})", sym, pos["qty"], price)
        return True

    def _log(self, side, sym, qty, price):
        self.state["log"].append(
            {"time": pd.Timestamp.now().isoformat(timespec="seconds"),
             "side": side, "symbol": sym, "qty": qty, "price": round(price, 2)}
        )
        print(f"{side:18}{sym:8}{qty:>6} @ {price:.2f}")
        notify(f"PAPER TRADE: {side} {qty} {sym} @ {price:.2f}")


# Skeleton for later: swap PaperBroker for Alpaca's paper API (pip install alpaca-py)
#   from alpaca.trading.client import TradingClient
#   client = TradingClient(API_KEY, SECRET, paper=True)
# then implement buy / sell_all / equity with client.submit_order(...)


# ------------------------------------------------------------------ agent
def run_agent(symbols: list, cfg: Config):
    broker = PaperBroker(cfg)
    # always manage stocks we already hold, even if not in today's list
    symbols = list(dict.fromkeys(symbols + list(broker.state["positions"])))
    prices = {}
    decisions = {}

    for sym in symbols:
        feats = make_features(load_prices(sym))
        valid = feats.dropna(subset=FEATURES)
        model = fit_model(valid.iloc[:-1])
        prob = float(model.predict_proba(valid.iloc[[-1]])[0, 1])
        prices[sym] = float(valid["close"].iloc[-1])
        decisions[sym] = prob
        print(f"{sym}: price {prices[sym]:.2f}, P(up tomorrow) = {prob:.2f}")

    equity = broker.equity(prices)
    print(f"Equity before trading: {equity:,.2f}\n")

    ranked = sorted(decisions.items(), key=lambda kv: kv[1], reverse=True)
    print("Ranking (most likely to rise first):")
    for i, (sym, prob) in enumerate(ranked, 1):
        tag = "  <- held" if sym in broker.state["positions"] else ""
        print(f"  {i}. {sym:6} P(up) = {prob:.2f}{tag}")
    print()

    trades_before = len(broker.state["log"])

    # 1) sells first, which frees up cash
    for sym, prob in ranked:
        held = broker.state["positions"].get(sym)
        if not held:
            continue
        price = prices[sym]
        if price / held["avg_price"] - 1 <= -cfg.stop_loss_pct:
            broker.sell_all(sym, price, "stop-loss")
        elif prob < cfg.sell_threshold:
            broker.sell_all(sym, price, "signal")

    # 2) buys: best-ranked first, up to max_positions
    for sym, prob in ranked:
        if len(broker.state["positions"]) >= cfg.max_positions:
            break
        if sym in broker.state["positions"] or prob <= cfg.buy_threshold:
            continue
        price = prices[sym]
        budget = min(equity * cfg.max_position_pct, broker.state["cash"])
        qty = int(budget // (price * (1 + cfg.cost_bps / 10_000)))
        broker.buy(sym, price, qty)

    broker.save()
    final = broker.equity(prices)
    print(f"\nEquity now: {final:,.2f}  |  cash: {broker.state['cash']:,.2f}")

    top3 = ", ".join(f"{s} {p:.2f}" for s, p in ranked[:3])
    n_trades = len(broker.state["log"]) - trades_before
    notify(f"Daily summary: equity {final:,.2f}, {n_trades} trade(s). Top picks: {top3}")


def show_status(cfg: Config):
    broker = PaperBroker(cfg)
    print(f"Cash: {broker.state['cash']:,.2f}")
    for sym, p in broker.state["positions"].items():
        print(f"  {sym}: {p['qty']} shares @ avg {p['avg_price']:.2f}")
    print("Last trades:")
    for t in broker.state["log"][-10:]:
        print(f"  {t['time']}  {t['side']:18}{t['symbol']:8}{t['qty']:>6} @ {t['price']}")


# -------------------------------------------------------------------- cli
if __name__ == "__main__":
    cfg = Config()
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "backtest":
        for s in args[1:]:
            backtest(s.upper(), cfg)
    elif len(args) >= 2 and args[0] == "run":
        run_agent([s.upper() for s in args[1:]], cfg)
    elif args and args[0] == "status":
        show_status(cfg)
    else:
        print(__doc__)
