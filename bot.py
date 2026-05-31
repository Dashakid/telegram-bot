import asyncio
import logging
import re
from typing import Dict, Optional, Set

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from analyst import GeminiAnalyst, build_pros_cons
from config import SETTINGS
from formatter import format_stock_alert, format_technicals
from fundamental import FMPClient
from news import NewsClient
from scheduler import create_daily_scheduler
from screener import analyze_ticker, get_universe_tickers
from store import SubscriberStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram-stock-bot")

TICKER_REGEX = re.compile(r"\b[A-Z]{1,5}\b")

# Curated list used exclusively for /demo — well-known tickers with reliable FMP
# data coverage. Kept at 20 so the command completes in ~30 seconds.
_DEMO_TICKERS = [
    "NVDA", "AAPL", "MSFT", "META", "GOOGL",
    "AMZN", "TSLA", "AMD",  "CRM",  "SNOW",
    "PLTR", "NET",  "DDOG", "CRWD", "ZS",
    "MDB",  "SHOP", "UBER", "COIN", "SMCI",
]
# Shorter inter-ticker delay for demo: 20 tickers × 7 calls = 140 FMP calls total,
# well within rate limits even at 0.3 s between tickers.
_DEMO_INTER_TICKER_DELAY = 0.3


def _extract_ticker(text: str) -> Optional[str]:
    matches = TICKER_REGEX.findall(text.upper())
    return matches[0] if matches else None


def _fallback_summary(ticker: str, passed: bool, failed_checks: list[str]) -> str:
    if passed:
        return (
            f"{ticker} passes this bot's current screening rules. "
            "Review technicals and position sizing before acting."
        )

    if failed_checks:
        top_reasons = "; ".join(failed_checks[:2])
        return (
            f"{ticker} does not currently meet this bot's strict screening rules. "
            f"Main gaps: {top_reasons}."
        )

    return (
        f"{ticker} has limited data right now, so no high-confidence summary could be built. "
        "Try again later or use /technicals for a quick market read."
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        store: SubscriberStore = context.bot_data["subscriber_store"]
        store.add(update.effective_chat.id)

    await update.message.reply_text(
        "I scan US stocks daily for strong revenue growth + catalyst news + margin quality.\n\n"
        "Commands:\n"
        "/analyse TICKER - Full pros/cons breakdown\n"
        "/technicals TICKER - RSI, MACD, Bollinger Bands\n"
        "/demo - Best signal from 20 tickers (great for demos)\n"
        "/test [TICKER] - Send a demo signal (default: NVDA)\n"
        "/scan10 - Dry run scanner on 10 tickers\n"
        "/stop - Unsubscribe from daily alerts\n"
        "You can also ask free-text questions about stocks."
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        store: SubscriberStore = context.bot_data["subscriber_store"]
        store.remove(update.effective_chat.id)
    await update.message.reply_text(
        "You've been unsubscribed from daily stock alerts. "
        "Type /start anytime to resubscribe."
    )


async def analyse_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        store: SubscriberStore = context.bot_data["subscriber_store"]
        store.add(update.effective_chat.id)

    if not context.args:
        await update.message.reply_text("Usage: /analyse TICKER")
        return

    ticker = context.args[0].upper().strip()
    if not re.fullmatch(r"[A-Z]{1,5}", ticker):
        await update.message.reply_text("Ticker must be 1-5 letters, e.g. /analyse AAPL")
        return

    fmp: FMPClient = context.bot_data["fmp_client"]
    news_client: NewsClient = context.bot_data["news_client"]
    gemini: GeminiAnalyst = context.bot_data["gemini_analyst"]

    # Attempt analysis; proceed even if FMP is unavailable (fallback to local logic)
    result = await asyncio.to_thread(analyze_ticker, ticker, fmp, news_client)
    
    pros, cons, checklist = build_pros_cons(result.snapshot, result.news_signals)

    if not pros:
        pros.append("No strong positives from the configured screen right now")
    if not cons:
        if result.failed_checks:
            cons.extend([f"Did not meet: {check}" for check in result.failed_checks[:3]])
        else:
            cons.append("No major red flags from configured risk checks")

    summary = await asyncio.to_thread(
        gemini.generate_summary,
        ticker,
        result.snapshot,
        result.news_items,
    )
    if not summary:
        summary = _fallback_summary(ticker, result.passed, result.failed_checks)

    price = result.snapshot.get("price")
    catalyst = _format_catalyst(result.news_signals)
    message = format_stock_alert(ticker, pros, cons, checklist, summary, price=price, catalyst=catalyst)
    await update.message.reply_text(message)


async def technicals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        store: SubscriberStore = context.bot_data["subscriber_store"]
        store.add(update.effective_chat.id)

    if not context.args:
        await update.message.reply_text("Usage: /technicals TICKER")
        return

    ticker = context.args[0].upper().strip()
    fmp: FMPClient = context.bot_data["fmp_client"]
    technicals = await asyncio.to_thread(fmp.get_technicals, ticker)

    if not technicals:
        await update.message.reply_text(f"No technical data available for {ticker}.")
        return

    await update.message.reply_text(format_technicals(ticker, technicals))


async def free_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    if update.effective_chat:
        store: SubscriberStore = context.bot_data["subscriber_store"]
        store.add(update.effective_chat.id)

    prompt = update.message.text.strip()
    ticker = _extract_ticker(prompt)

    fmp: FMPClient = context.bot_data["fmp_client"]
    news_client: NewsClient = context.bot_data["news_client"]
    gemini: GeminiAnalyst = context.bot_data["gemini_analyst"]

    stock_context: Dict = {}
    if ticker:
        result = await asyncio.to_thread(analyze_ticker, ticker, fmp, news_client)
        stock_context = {
            "ticker": ticker,
            "snapshot": result.snapshot,
            "news_signals": result.news_signals,
        }

    answer = await asyncio.to_thread(gemini.answer_question, prompt, stock_context)
    await update.message.reply_text(answer)


def _score_result(result, pros: list, cons: list) -> tuple:
    """Return a comparable score tuple for ranking demo candidates.

    Higher is better on every component:
      (passed, #pros, revenue_growth_capped, -#cons)
    """
    growth = result.snapshot.get("latest_revenue_growth") or 0.0
    return (
        int(result.passed),        # 1 if passes screen, 0 if not
        len(pros),                 # more positives = better
        min(growth, 150.0),        # growth magnitude, capped to avoid outlier distortion
        -len(cons),                # fewer negatives = better
    )


def _format_catalyst(news_signals: Dict) -> Optional[str]:
    """Return a human-readable catalyst string from matched news keywords, or None."""
    keywords = news_signals.get("matched_keywords", [])
    if not keywords:
        return None
    return ", ".join(keywords)


def _resolve_chat_ids(store: SubscriberStore) -> Set[int]:
    chat_ids = store.all()
    if SETTINGS.default_chat_id:
        try:
            chat_ids.add(int(SETTINGS.default_chat_id))
        except ValueError:
            logger.error("Invalid TELEGRAM_CHAT_ID in env: %s", SETTINGS.default_chat_id)
    return chat_ids


async def _run_scan_for_tickers(application: Application, tickers: list[str], chat_ids: Set[int]) -> int:
    fmp: FMPClient = application.bot_data["fmp_client"]
    news_client: NewsClient = application.bot_data["news_client"]
    gemini: GeminiAnalyst = application.bot_data["gemini_analyst"]

    matches = 0
    for i, ticker in enumerate(tickers):
        try:
            result = await asyncio.to_thread(analyze_ticker, ticker, fmp, news_client)
            if not result.passed:
                continue

            pros, cons, checklist = build_pros_cons(result.snapshot, result.news_signals)
            summary = await asyncio.to_thread(
                gemini.generate_summary,
                ticker,
                result.snapshot,
                result.news_items,
            )
            if not summary:
                logger.warning("Gemini summary unavailable for %s — using fallback", ticker)
                summary = _fallback_summary(ticker, result.passed, result.failed_checks)

            price = result.snapshot.get("price")
            catalyst = _format_catalyst(result.news_signals)
            message = format_stock_alert(
                ticker, pros, cons, checklist, summary,
                price=price, catalyst=catalyst,
            )
            for chat_id in chat_ids:
                await application.bot.send_message(chat_id=chat_id, text=message)
            matches += 1
        except Exception:
            logger.exception("Unhandled error during scan for ticker %s", ticker)

        # Throttle: prevents bursting FMP rate limits across the full scan.
        # 1.5s between tickers ≈ 7 FMP calls per 2.5s ≈ 170 calls/min (Starter plan: 300/min).
        if i < len(tickers) - 1:
            await asyncio.sleep(SETTINGS.scan_inter_ticker_delay)

    return matches


async def _notify_all(application: Application, chat_ids: Set[int], text: str) -> None:
    for chat_id in chat_ids:
        try:
            await application.bot.send_message(chat_id=chat_id, text=text)
        except Exception:
            logger.exception("Failed to send notification to chat_id=%s", chat_id)


async def run_daily_scan(application: Application) -> None:
    store: SubscriberStore = application.bot_data["subscriber_store"]
    chat_ids = _resolve_chat_ids(store)

    if not chat_ids:
        logger.warning("No subscribed chat IDs found. Daily scan skipped.")
        return

    fmp: FMPClient = application.bot_data["fmp_client"]
    news_client: NewsClient = application.bot_data["news_client"]

    # Probe both data sources before starting — fail loudly if either is unreachable.
    fmp_ok = await asyncio.to_thread(fmp.health_check)
    if not fmp_ok:
        msg = (
            "⚠️ Daily scan aborted — Financial Modeling Prep (FMP) is unreachable.\n"
            "Fundamental data and screening will not work until FMP recovers."
        )
        logger.error(msg)
        await _notify_all(application, chat_ids, msg)
        return

    finnhub_ok = await asyncio.to_thread(news_client.health_check)
    if not finnhub_ok:
        msg = (
            "⚠️ Finnhub is unreachable — daily scan will use FMP news as fallback.\n"
            "Results may be less timely than usual."
        )
        logger.warning(msg)
        await _notify_all(application, chat_ids, msg)
        # Not a hard abort — FMP news fallback is functional.

    tickers = await asyncio.to_thread(get_universe_tickers, 500)
    logger.info("Starting daily scan for %s tickers", len(tickers))
    matches = await _run_scan_for_tickers(application, tickers, chat_ids)
    logger.info("Daily scan complete. Alerts sent for %s stocks.", matches)

    # Always send a daily completion update so subscribers know the scan ran.
    if matches == 0:
        await _notify_all(
            application,
            chat_ids,
            "📭 Daily scan complete: no stocks met the full criteria today.",
        )
    else:
        await _notify_all(
            application,
            chat_ids,
            f"✅ Daily scan complete: {matches} stock(s) met the criteria today.",
        )


async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a single demo signal for a ticker so the full alert format can be verified."""
    if update.effective_chat:
        store: SubscriberStore = context.bot_data["subscriber_store"]
        store.add(update.effective_chat.id)

    ticker = "NVDA"
    if context.args:
        raw = context.args[0].upper().strip()
        if re.fullmatch(r"[A-Z]{1,5}", raw):
            ticker = raw
        else:
            await update.message.reply_text("Invalid ticker. Usage: /test or /test AAPL")
            return

    await update.message.reply_text(f"Running demo signal for ${ticker}...")

    fmp: FMPClient = context.bot_data["fmp_client"]
    news_client: NewsClient = context.bot_data["news_client"]
    gemini: GeminiAnalyst = context.bot_data["gemini_analyst"]

    result = await asyncio.to_thread(analyze_ticker, ticker, fmp, news_client)
    pros, cons, checklist = build_pros_cons(result.snapshot, result.news_signals)

    if not pros:
        pros.append("No strong positives from the configured screen right now")
    if not cons:
        if result.failed_checks:
            cons.extend([f"Did not meet: {c}" for c in result.failed_checks[:3]])
        else:
            cons.append("No major red flags from configured risk checks")

    summary = await asyncio.to_thread(
        gemini.generate_summary, ticker, result.snapshot, result.news_items
    )
    if not summary:
        summary = _fallback_summary(ticker, result.passed, result.failed_checks)

    price = result.snapshot.get("price")
    catalyst = _format_catalyst(result.news_signals)
    screen_status = "PASSES screen ✅" if result.passed else "Does NOT pass screen ❌"
    message = format_stock_alert(
        ticker, pros, cons, checklist, summary,
        price=price, catalyst=catalyst, is_demo=True,
    )
    await update.message.reply_text(f"{message}\n\n[Screen result: {screen_status}]")


async def demo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan 20 curated tickers and return the single best signal as a [DEMO] alert.

    Uses the most recent available fundamental data so it works outside market hours.
    Always returns a result — if no ticker passes the full screen, the highest-scoring
    candidate is returned so the demo never comes up empty.
    """
    if update.effective_chat:
        store: SubscriberStore = context.bot_data["subscriber_store"]
        store.add(update.effective_chat.id)

    await update.message.reply_text(
        f"Scanning {len(_DEMO_TICKERS)} tickers for the best demo signal… "
        "(~30 seconds)"
    )

    fmp: FMPClient = context.bot_data["fmp_client"]
    news_client: NewsClient = context.bot_data["news_client"]
    gemini: GeminiAnalyst = context.bot_data["gemini_analyst"]

    best_result = None
    best_pros: list[str] = []
    best_cons: list[str] = []
    best_checklist: dict = {}
    best_score: tuple = (-1,)

    for i, ticker in enumerate(_DEMO_TICKERS):
        try:
            result = await asyncio.to_thread(analyze_ticker, ticker, fmp, news_client)
            pros, cons, checklist = build_pros_cons(result.snapshot, result.news_signals)
            score = _score_result(result, pros, cons)
            if score > best_score:
                best_score = score
                best_result = result
                best_pros = pros
                best_cons = cons
                best_checklist = checklist
        except Exception:
            logger.exception("Demo scan error for ticker %s", ticker)

        if i < len(_DEMO_TICKERS) - 1:
            await asyncio.sleep(_DEMO_INTER_TICKER_DELAY)

    if best_result is None:
        await update.message.reply_text(
            "⚠️ Could not retrieve data for any ticker. "
            "Check that FMP is reachable (/test AAPL to diagnose)."
        )
        return

    ticker = best_result.ticker
    if not best_pros:
        best_pros.append("No strong positives from the configured screen right now")
    if not best_cons:
        if best_result.failed_checks:
            best_cons.extend([f"Did not meet: {c}" for c in best_result.failed_checks[:3]])
        else:
            best_cons.append("No major red flags from configured risk checks")

    summary = await asyncio.to_thread(
        gemini.generate_summary, ticker, best_result.snapshot, best_result.news_items
    )
    if not summary:
        summary = _fallback_summary(ticker, best_result.passed, best_result.failed_checks)

    price = best_result.snapshot.get("price")
    catalyst = _format_catalyst(best_result.news_signals)
    screen_label = "Passes screen ✅" if best_result.passed else "Best candidate (did not pass screen) ❌"
    message = format_stock_alert(
        ticker, best_pros, best_cons, best_checklist, summary,
        price=price, catalyst=catalyst, is_demo=True,
    )
    header = (
        "📊 Best signal from today's scan of 20 large-cap tickers. "
        "Criteria: revenue growth + news catalyst + margin quality."
    )
    await update.message.reply_text(f"{header}\n\n{message}\n\n[{screen_label}]")


async def scan10_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        store: SubscriberStore = context.bot_data["subscriber_store"]
        store.add(update.effective_chat.id)

    chat_ids = _resolve_chat_ids(context.bot_data["subscriber_store"])
    if not chat_ids:
        await update.message.reply_text("No chat subscriptions available for dry run.")
        return

    await update.message.reply_text("Running dry scan for 10 tickers...")
    tickers = await asyncio.to_thread(get_universe_tickers, 10)
    matches = await _run_scan_for_tickers(context.application, tickers, chat_ids)
    await update.message.reply_text(f"Dry scan complete. Alerts sent for {matches} stocks.")


async def post_init(application: Application) -> None:
    async def scheduled_job() -> None:
        await run_daily_scan(application)

    scheduler = create_daily_scheduler(scheduled_job)
    scheduler.start()
    application.bot_data["scheduler"] = scheduler
    logger.info("Scheduler started for weekday 07:00 ET.")

    # Write sentinel file so the Docker health check can confirm the bot is running.
    try:
        with open("/tmp/bot.healthy", "w") as f:
            f.write("ok")
    except OSError:
        logger.warning("Could not write health sentinel file /tmp/bot.healthy")


async def post_shutdown(application: Application) -> None:
    scheduler = application.bot_data.get("scheduler")
    if scheduler:
        scheduler.shutdown(wait=False)

    store: SubscriberStore | None = application.bot_data.get("subscriber_store")
    if store:
        store.close()


def validate_env() -> None:
    _REQUIRED: list[tuple[str, str, str]] = [
        (
            "TELEGRAM_BOT_TOKEN",
            SETTINGS.telegram_bot_token,
            "Create a bot via @BotFather on Telegram (https://t.me/BotFather)",
        ),
        (
            "FMP_API_KEY",
            SETTINGS.fmp_api_key,
            "Register at https://financialmodelingprep.com — required for all screening and fundamentals",
        ),
        (
            "FINNHUB_API_KEY",
            SETTINGS.finnhub_api_key,
            "Register at https://finnhub.io — required for news data",
        ),
        (
            "GEMINI_API_KEY",
            SETTINGS.gemini_api_key,
            "Generate a key at https://aistudio.google.com/app/apikey — required for AI summaries",
        ),
    ]

    errors = [
        f"  • {name}: {hint}"
        for name, value, hint in _REQUIRED
        if not value
    ]

    if errors:
        raise RuntimeError(
            "Bot cannot start — the following required environment variables are not set:\n"
            + "\n".join(errors)
            + "\n\nCopy .env.example to .env and fill in every value."
        )


def main() -> None:
    validate_env()

    subscriber_store = SubscriberStore(SETTINGS.subscriber_db_path)
    fmp_client = FMPClient(SETTINGS.fmp_api_key)
    news_client = NewsClient(SETTINGS.finnhub_api_key, fmp_client)
    gemini_analyst = GeminiAnalyst(SETTINGS.gemini_api_key)

    app = (
        Application.builder()
        .token(SETTINGS.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.bot_data["subscriber_store"] = subscriber_store
    app.bot_data["fmp_client"] = fmp_client
    app.bot_data["news_client"] = news_client
    app.bot_data["gemini_analyst"] = gemini_analyst

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("analyse", analyse_command))
    app.add_handler(CommandHandler("technicals", technicals_command))
    app.add_handler(CommandHandler("test", test_command))
    app.add_handler(CommandHandler("demo", demo_command))
    app.add_handler(CommandHandler("scan10", scan10_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_text_handler))

    logger.info("Bot started and polling.")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
