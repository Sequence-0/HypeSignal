"""Interactive Telegram Authentication Helper to generate TELEGRAM_SESSION_STRING for .env."""

from __future__ import annotations

import asyncio
import os
import sys

from telethon import TelegramClient
from telethon.sessions import StringSession

from hypesignal.config import find_dotenv, load_env


async def main() -> None:
    """Run interactive Telegram authentication flow."""
    load_env()

    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    test_mode = os.getenv("TELEGRAM_TEST_MODE", "false").lower() in ("1", "true", "yes")

    print("=" * 60)
    print("HypeSignal Telegram MTProto Session Authenticator")
    print("=" * 60)

    if not api_id or not api_hash:
        print("\n[ERROR] TELEGRAM_API_ID or TELEGRAM_API_HASH is missing in .env.")
        print("Please configure them first in your .env file.")
        sys.exit(1)

    print("\nConfiguration detected:")
    print(f"  API ID:    {api_id}")
    print(f"  API Hash:  {api_hash[:4]}...{api_hash[-4:]}")
    print(f"  Test Mode: {test_mode}")

    session = StringSession()
    if test_mode:
        dc_id = int(os.getenv("TELEGRAM_TEST_DC_ID", "2"))
        dc_ip = os.getenv("TELEGRAM_TEST_DC_IP", "149.154.167.40")
        dc_port = int(os.getenv("TELEGRAM_TEST_DC_PORT", "443"))
        print(f"  Target DC: {dc_id} ({dc_ip}:{dc_port})")
        session.set_dc(dc_id, dc_ip, dc_port)

    client = TelegramClient(
        session=session,
        api_id=int(api_id),
        api_hash=api_hash,
        device_model=os.getenv("TELEGRAM_APP_TITLE", "HypeSignal"),
        app_version="1.0.0",
        system_version="Linux",
    )

    await client.connect()

    print("\nAuthentication Options:")
    print("  1) Telegram User Account (Phone number + verification code)")
    print("  2) Telegram Bot Token (from @BotFather)")
    choice = input("\nSelect authentication method [1/2] (default: 1): ").strip() or "1"

    if choice == "2":
        bot_token = input("\nEnter Bot Token: ").strip()
        if not bot_token:
            print("No bot token provided. Exiting.")
            sys.exit(1)
        await client.start(bot_token=bot_token)
        session_str = client.session.save()
        print("\n[SUCCESS] Bot authorized successfully!")
    else:
        print("\nNote: For MTProto test environment, test numbers are 99966XYYYY (e.g. 9996611111).")
        print("Verification code for test numbers on test servers is always 22222.")
        phone = input("Enter phone number with country code (e.g. +1234567890): ").strip()
        if not phone:
            print("No phone number provided. Exiting.")
            sys.exit(1)

        await client.start(phone=phone)
        session_str = client.session.save()
        print("\n[SUCCESS] User account authorized successfully!")

    print("\n" + "=" * 60)
    print("Generated TELEGRAM_SESSION_STRING:")
    print("=" * 60)
    print(session_str)
    print("=" * 60)

    dotenv_path = find_dotenv()
    if dotenv_path and os.path.exists(dotenv_path):
        ans = input(f"\nWould you like to automatically append this to {dotenv_path}? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            with open(dotenv_path, "a", encoding="utf-8") as f:
                f.write(f"\nTELEGRAM_SESSION_STRING={session_str}\n")
            print(f"Updated {dotenv_path} successfully!")
    else:
        print("\nAdd this line to your .env file:")
        print(f"TELEGRAM_SESSION_STRING={session_str}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
