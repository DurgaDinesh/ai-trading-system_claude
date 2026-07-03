"""
Run this script once in your own terminal to generate your trading PIN hash.
The hash gets pasted into config/.env as TRADE_PIN_HASH=...

Usage: python generate_pin.py
"""
import getpass
import bcrypt

def main():
    print("\n=== NiftySniper PIN Setup ===\n")
    while True:
        pin = getpass.getpass("Enter your 6-digit trading PIN (hidden): ").strip()
        if not pin.isdigit() or len(pin) != 6:
            print("ERROR: PIN must be exactly 6 digits. Try again.\n")
            continue
        confirm = getpass.getpass("Confirm PIN: ").strip()
        if pin != confirm:
            print("ERROR: PINs do not match. Try again.\n")
            continue
        break

    pin_hash = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
    print("\n" + "=" * 60)
    print("Copy this line into config/.env:")
    print("=" * 60)
    print(f"\nTRADE_PIN_HASH={pin_hash}\n")
    print("=" * 60)
    print("Done. Keep your PIN safe — it authorizes every live trade.")

if __name__ == "__main__":
    main()
