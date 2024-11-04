"""Script to automatically login each website"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from playwright.sync_api import sync_playwright
from browser_env.env_config import ACCOUNTS

DATASET = os.environ["DATASET"]
if DATASET == "webarena":
    from browser_env.env_config import (
        GITLAB,
        REDDIT,
        SHOPPING,
        SHOPPING_ADMIN,
    )
    SITES = ["gitlab", "shopping", "shopping_admin", "reddit"]
    URLS = [
        f"{GITLAB}/-/profile",
        f"{SHOPPING}/wishlist/",
        f"{SHOPPING_ADMIN}/dashboard",
        f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
    ]
    EXACT_MATCH = [True, True, True, True]
    KEYWORDS = ["", "", "Dashboard", "Delete"]

elif DATASET == "visualwebarena":
    from browser_env.env_config import (
        CLASSIFIEDS,
        REDDIT,
        SHOPPING,
    )
    SITES = ["shopping", "reddit", "classifieds"]
    URLS = [
        f"{SHOPPING}/wishlist/",
        f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
        f"{CLASSIFIEDS}/index.php?page=user&action=items",
    ]
    EXACT_MATCH = [True, True, True]
    KEYWORDS = ["", "Delete", "My listings"]
else:
    raise ValueError(f"Dataset not implemented: {DATASET}")

HEADLESS = True
SLOW_MO = 0

assert len(SITES) == len(URLS) == len(EXACT_MATCH) == len(KEYWORDS)

def is_expired(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True, retries: int = 3
) -> bool:
    """Test whether the cookie is expired with retry logic."""
    if not storage_state.exists():
        return True

    for attempt in range(retries):
        context_manager = sync_playwright()
        playwright = context_manager.__enter__()
        browser = playwright.chromium.launch(headless=True, slow_mo=SLOW_MO)
        context = browser.new_context(storage_state=storage_state)
        page = context.new_page()
        page.goto(url)
        time.sleep(5)
        d_url = page.url
        content = page.content()
        context_manager.__exit__()

        # Check content for keyword match or URL match
        if keyword and keyword in content:
            return False  # Cookie is valid
        elif not keyword:
            if (url_exact and d_url.replace("localhost", "127.0.0.1") == url.replace("localhost", "127.0.0.1")) \
            or (not url_exact and url.replace("localhost", "127.0.0.1") in d_url.replace("localhost", "127.0.0.1")):
                return False  # Cookie is valid

        # If we've reached this point, retry the check
        time.sleep(5)  # Add a short delay before retrying
        print("Retrying......")

    return True  # Cookie is expired after retries


def renew_comb(comb: list[str], auth_folder: str = "./.auth") -> None:
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context()
    page = context.new_page()

    if "shopping" in comb:
        username = ACCOUNTS["shopping"]["username"]
        password = ACCOUNTS["shopping"]["password"]
        page.goto(f"{SHOPPING}/customer/account/login/")
        page.get_by_label("Email", exact=True).fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Sign In").click()

    if "reddit" in comb:
        username = ACCOUNTS["reddit"]["username"]
        password = ACCOUNTS["reddit"]["password"]
        page.goto(f"{REDDIT}/login")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password").fill(password)
        page.get_by_role("button", name="Log in").click()

    if "classifieds" in comb:
        username = ACCOUNTS["classifieds"]["username"]
        password = ACCOUNTS["classifieds"]["password"]
        page.goto(f"{CLASSIFIEDS}/index.php?page=login")
        page.locator("#email").fill(username)
        page.locator("#password").fill(password)
        page.get_by_role("button", name="Log in").click()

    if "shopping_admin" in comb:
        username = ACCOUNTS["shopping_admin"]["username"]
        password = ACCOUNTS["shopping_admin"]["password"]
        page.goto(f"{SHOPPING_ADMIN}")
        page.get_by_placeholder("user name").fill(username)
        page.get_by_placeholder("password").fill(password)
        page.get_by_role("button", name="Sign in").click()

    if "gitlab" in comb:
        username = ACCOUNTS["gitlab"]["username"]
        password = ACCOUNTS["gitlab"]["password"]
        page.goto(f"{GITLAB}/users/sign_in")
        page.get_by_test_id("username-field").click()
        page.get_by_test_id("username-field").fill(username)
        page.get_by_test_id("username-field").press("Tab")
        page.get_by_test_id("password-field").fill(password)
        page.get_by_test_id("sign-in-button").click()

    context.storage_state(path=f"{auth_folder}/{'.'.join(comb)}_state.json")

    context_manager.__exit__()


def get_site_comb_from_filepath(file_path: str) -> list[str]:
    comb = os.path.basename(file_path).rsplit("_", 1)[0].split(".")
    return comb


def main(auth_folder: str = "./.auth") -> None:
    pairs = list(combinations(SITES, 2))

    with ThreadPoolExecutor(max_workers=8) as executor:
        for pair in pairs:
            # Auth doesn't work on this pair as they share the same cookie
            if "reddit" in pair and (
                "shopping" in pair or "shopping_admin" in pair
            ):
                continue
            executor.submit(
                renew_comb, list(sorted(pair)), auth_folder=auth_folder
            )

        for site in SITES:
            executor.submit(renew_comb, [site], auth_folder=auth_folder)
    
    # parallel checking if the cookies are expired  
    futures = []
    cookie_files = list(glob.glob(f"{auth_folder}/*.json"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        for c_file in cookie_files:
            comb = get_site_comb_from_filepath(c_file)
            for cur_site in comb:
                url = URLS[SITES.index(cur_site)]
                keyword = KEYWORDS[SITES.index(cur_site)]
                match = EXACT_MATCH[SITES.index(cur_site)]
                future = executor.submit(
                    is_expired, Path(c_file), url, keyword, match
                )
                futures.append(future)

    for i, future in enumerate(futures):
        assert not future.result(), f"Cookie {cookie_files[i]} expired."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_list", nargs="+", default=[])
    parser.add_argument("--auth_folder", type=str, default="./.auth")
    args = parser.parse_args()
    if not args.site_list:
        main()
    else:
        renew_comb(args.site_list, auth_folder=args.auth_folder)