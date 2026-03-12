#!/usr/bin/env python3
"""
Simple Python examples in one file:
- Console I/O (input/print)
- Variables & basic types
- Lists, dicts, sets
- Loops and comprehensions
- Functions (with type hints)
- Error handling (try/except)
- File I/O (read/write) with encoding="utf-8"
- A tiny class
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from termcolor import colored


# Major variables (ALL CAPS)
APP_NAME = "code_prompting"
OUTPUT_FILE = Path("example_output.txt")
DEFAULT_NAME = "Student"
DEFAULT_NUMBERS = [3, 1, 4, 1, 5]


def status(msg: str, color: str = "cyan") -> None:
    print(colored(f"[{APP_NAME}] {msg}", color))


def safe_input(prompt: str, default: str) -> str:
    """
    Reads user input when interactive; otherwise returns a default.
    """
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            value = input(prompt).strip()
            return value if value else default
        return default
    except Exception as e:
        status(f"Input failed, using default. Error: {e}", "yellow")
        return default


def summarize_numbers(nums: list[int]) -> dict[str, float]:
    if not nums:
        raise ValueError("nums must not be empty")
    total = sum(nums)
    return {
        "count": float(len(nums)),
        "sum": float(total),
        "min": float(min(nums)),
        "max": float(max(nums)),
        "avg": float(total / len(nums)),
    }


def square_all(nums: list[int]) -> list[int]:
    return [n * n for n in nums]


@dataclass(frozen=True)
class Person:
    name: str

    def greet(self) -> str:
        return f"Hello, {self.name}!"


def write_report(path: Path, lines: list[str]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to write report to {path}") from e


def read_report(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read report from {path}") from e


def main() -> int:
    status("Starting examples.")

    # 1) Console I/O + variables
    name = safe_input("Your name (press Enter for default): ", DEFAULT_NAME)
    age_text = safe_input("Your age (number): ", "20")
    try:
        age = int(age_text)
    except Exception as e:
        status(f"Age was not a valid int; defaulting to 20. Error: {e}", "yellow")
        age = 20

    status(f"Name={name!r}, Age={age}", "green")

    # 2) List + loop + sorting
    numbers = DEFAULT_NUMBERS.copy()
    numbers.append(age)
    status(f"Numbers list: {numbers}", "cyan")

    numbers_sorted = sorted(numbers)
    status(f"Sorted: {numbers_sorted}", "cyan")

    # 3) Dict + set
    profile = {"name": name, "age": age, "favorite_numbers": numbers_sorted}
    unique_numbers = set(numbers_sorted)
    status(f"Profile dict keys: {list(profile.keys())}", "cyan")
    status(f"Unique numbers (set): {sorted(unique_numbers)}", "cyan")

    # 4) Functions + error handling
    try:
        stats = summarize_numbers(numbers_sorted)
        squares = square_all(numbers_sorted)
    except Exception as e:
        status(f"Computation failed. Error: {e}", "red")
        return 1

    status(f"Stats: {stats}", "green")
    status(f"Squares (list comprehension): {squares}", "green")

    # 5) Tiny class example
    person = Person(name=name)
    status(person.greet(), "magenta")

    # 6) File I/O (write + read)
    report_lines = [
        f"Name: {profile['name']}",
        f"Age: {profile['age']}",
        f"Numbers: {profile['favorite_numbers']}",
        f"Stats: {stats}",
        f"Squares: {squares}",
    ]
    status(f"Writing report to {OUTPUT_FILE.resolve()}", "cyan")
    try:
        write_report(OUTPUT_FILE, report_lines)
        content = read_report(OUTPUT_FILE)
    except Exception as e:
        status(f"File I/O failed. Error: {e}", "red")
        return 1

    status("Read back report content:", "cyan")
    print(content)

    status("Done.", "green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
