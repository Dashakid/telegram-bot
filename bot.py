import asyncio
import logging
import re
from typing import Dict, Optional, Set

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from analyst import GeminiAnalyst, build_pros_cons
from config import SETTINGS
from formatter import format_stock_alert, format_technicals
from fundamental import FMPClient
from news import NewsClient
from scheduler import create_daily_scheduler
from screener import analyze_ticker, get_universe_tickers


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram-stock-bot")

SUBSCRIBED_CHAT_IDS: Set[int] = set()
TICKER_REGEX = re.compile(r"\b[A-Z]{1,5}\b")


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
        SUBSCRIBED_CHAT_IDS.add(update.effective_chat.id)

    await update.message.reply_text(
        "I scan US stocks daily for strong revenue growth + catalyst news + margin quality.\n\n"
        "Commands:\n"
        "/analyse TICKER - Full pros/cons breakdown\n"
        "/technicals TICKER - RSI, MACD, Bollinger Bands\n"
        "/scan10 - Dry run scanner on 10 tickers\n"
        "/stop - Unsubscribe from daily alerts\n"
        "You can also ask free-text questions about stocks."
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        SUBSCRIBED_CHAT_IDS.discard(update.effective_chat.id)
    await update.message.reply_text(
        "You've been unsubscribed from daily stock alerts. "
        "Type /start anytime to resubscribe."
    )


async def analyse_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        SUBSCRIBED_CHAT_IDS.add(update.effective_chat.id)

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

    message = format_stock_alert(ticker, pros, cons, checklist, summary)
    await update.message.reply_text(message)


async def technicals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        SUBSCRIBED_CHAT_IDS.add(update.effective_chat.id)

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
        SUBSCRIBED_CHAT_IDS.add(update.effective_chat.id)

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


def _resolve_chat_ids() -> Set[int]:
    chat_ids = set(SUBSCRIBED_CHAT_IDS)
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
    for ticker in tickers:
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
            message = format_stock_alert(ticker, pros, cons, checklist, summary)
            for chat_id in chat_ids:
                await application.bot.send_message(chat_id=chat_id, text=message)
            matches += 1
        except Exception:
            logger.exception("Unhandled error during scan for ticker %s", ticker)

    return matches


async def run_daily_scan(application: Application) -> None:
    chat_ids = _resolve_chat_ids()

    if not chat_ids:
        logger.warning("No subscribed chat IDs found. Daily scan skipped.")
        return

    tickers = await asyncio.to_thread(get_universe_tickers, 500)
    logger.info("Starting daily scan for %s tickers", len(tickers))
    matches = await _run_scan_for_tickers(application, tickers, chat_ids)
    logger.info("Daily scan complete. Alerts sent for %s stocks.", matches)


async def scan10_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        SUBSCRIBED_CHAT_IDS.add(update.effective_chat.id)

    chat_ids = _resolve_chat_ids()
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


async def post_shutdown(application: Application) -> None:
    scheduler = application.bot_data.get("scheduler")
    if scheduler:
        scheduler.shutdown(wait=False)


def validate_env() -> None:
    missing = []
    if not SETTINGS.telegram_bot_token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not SETTINGS.finnhub_api_key:
        missing.append("FINNHUB_API_KEY")
    if not SETTINGS.gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not SETTINGS.fmp_api_key:
        logger.warning("FMP_API_KEY not set — fundamental data and stock screening will be limited.")

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def main() -> None:
    load_dotenv()
    validate_env()

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

    app.bot_data["fmp_client"] = fmp_client
    app.bot_data["news_client"] = news_client
    app.bot_data["gemini_analyst"] = gemini_analyst

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("analyse", analyse_command))
    app.add_handler(CommandHandler("technicals", technicals_command))
    app.add_handler(CommandHandler("scan10", scan10_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_text_handler))

    logger.info("Bot started and polling.")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
