from __future__ import annotations

from app.runtime import init_runtime


def main() -> None:
    init_runtime()
    print("Finance Agent runtime initialized.")


if __name__ == "__main__":
    main()
